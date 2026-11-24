from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beam.upa_dft_sim import UPADFTCodebook, phase_only_precoder, simulate_position
from tools.position_beamforming_test import write_svg


ROOT = Path(__file__).resolve().parents[1]


class UPADFTCodebookTests(unittest.TestCase):
    def setUp(self):
        self.codebook = UPADFTCodebook(
            rows=4,
            columns=4,
            oversampling=4,
            rf_hz=7.4e9,
            element_spacing_m=18.5e-3,
            max_abs_azimuth_deg=60.0,
            max_abs_elevation_deg=60.0,
        )

    def test_weights_are_unit_norm_and_broadside_exists(self):
        norms = [np.vdot(self.codebook.weights(beam), self.codebook.weights(beam)).real for beam in self.codebook]
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-12))
        broadside = self.codebook.nearest_direction(0.0, 0.0)
        self.assertEqual((broadside.horizontal_bin, broadside.vertical_bin), (0, 0))

    def test_exact_codebook_direction_has_zero_quantization_loss(self):
        beam = next(item for item in self.codebook if item.horizontal_bin == 2 and item.vertical_bin == -1)
        steering = self.codebook.steering_vector(beam.u, beam.v)
        selected = self.codebook.nearest_direction(beam.u, beam.v)
        oracle, overlap = self.codebook.oracle_beam(steering)
        self.assertEqual(selected.index, beam.index)
        self.assertEqual(oracle.index, beam.index)
        self.assertAlmostEqual(overlap, 1.0, places=12)


class ThreeRUSimulationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(
            (ROOT / "config" / "position_beamforming_test.json").read_text(encoding="utf-8")
        )
        upa = self.cfg["upa"]
        dft = self.cfg["codebook"]
        self.codebook = UPADFTCodebook(
            rows=upa["rows"],
            columns=upa["columns"],
            oversampling=dft["oversampling"],
            rf_hz=self.cfg["frequency_hz"],
            element_spacing_m=upa["element_spacing_m"],
            max_abs_azimuth_deg=dft["max_abs_azimuth_deg"],
            max_abs_elevation_deg=dft["max_abs_elevation_deg"],
        )

    def test_position_selection_and_single_antenna_direct_sum(self):
        result = simulate_position(self.cfg["rus"], self.cfg["ue_position_m"], self.codebook)
        self.assertEqual(len(result["per_ru"]), 3)
        self.assertEqual(result["model"]["ue_antennas"], 1)
        for ru in result["per_ru"]:
            self.assertTrue(ru["selected_matches_codebook_oracle"])
            self.assertGreaterEqual(ru["selected_power_retained_vs_mrt"], 0.9)
            self.assertLessEqual(ru["selected_gain"], ru["mrt_gain"] * (1.0 + 1e-12))

        net = result["network"]
        self.assertEqual(net["ue_antennas"], 1)
        self.assertEqual(net["receive_combining"], "none_over_the_air_complex_superposition")
        self.assertAlmostEqual(net["direct_superposition_gain"], net["selected_raw_gain"], places=15)
        self.assertGreaterEqual(net["selected_coherent_equal_power_gain"], net["selected_noncoherent_gain"])
        self.assertLessEqual(net["selected_coherent_equal_power_gain"], net["network_mrt_gain"] * (1.0 + 1e-12))
        self.assertGreaterEqual(net["phase_aligned_dft_power_retained_vs_network_mrt"], 0.9)
        # With no RU-level phase correction, the default geometry causes
        # destructive over-the-air addition at the single UE antenna.
        self.assertLess(net["direct_superposition_gain"], net["best_single_ru_gain"])
        self.assertLess(net["direct_power_retained_vs_network_mrt"], 0.3)
        self.assertEqual(
            net["transmit_precoding"],
            "position_dft_plus_predicted_effective_csi_phase",
        )
        self.assertAlmostEqual(
            net["predicted_csi_precoded_gain"],
            net["selected_coherent_equal_power_gain"],
            places=15,
        )

    def test_phase_only_precoder_aligns_predicted_effective_channels(self):
        predicted = np.asarray([1.0j, -2.0, 3.0 - 3.0j], dtype=np.complex128)
        precoder = phase_only_precoder(predicted)
        aligned = predicted * precoder
        self.assertTrue(np.allclose(np.abs(precoder), 1.0, atol=1e-12))
        self.assertTrue(np.allclose(np.angle(aligned), np.angle(aligned[0]), atol=1e-12))

    def test_csi_phase_prediction_error_reduces_coherent_gain(self):
        exact = simulate_position(self.cfg["rus"], self.cfg["ue_position_m"], self.codebook)
        perturbed_rus = [dict(ru) for ru in self.cfg["rus"]]
        for ru, error_deg in zip(perturbed_rus, [0.0, 30.0, -30.0]):
            ru["predicted_csi_phase_error_deg"] = error_deg
        perturbed = simulate_position(perturbed_rus, self.cfg["ue_position_m"], self.codebook)
        exact_net = exact["network"]
        perturbed_net = perturbed["network"]
        self.assertLess(
            perturbed_net["predicted_csi_precoded_gain"],
            exact_net["predicted_csi_precoded_gain"],
        )
        self.assertLess(
            perturbed_net["predicted_csi_precoding_loss_vs_perfect_phase_db"],
            0.0,
        )

    def test_direct_sum_uses_complex_channels_without_phase_correction(self):
        result = simulate_position(self.cfg["rus"], self.cfg["ue_position_m"], self.codebook)
        net = result["network"]
        # Exact 7.4-GHz path phases can add or cancel.  Explicit phase
        # compensation must therefore be reported separately from the raw sum.
        self.assertGreaterEqual(
            net["selected_coherent_equal_power_gain"],
            net["direct_superposition_gain"] - 1e-18,
        )
        scalars = np.asarray([
            complex(ru["selected_channel_real"], ru["selected_channel_imag"])
            for ru in result["per_ru"]
        ])
        corrections = np.exp(1j * np.radians([
            ru["inter_ru_phase_correction_deg"] for ru in result["per_ru"]
        ]))
        aligned_gain = abs(np.sum(scalars * corrections) / np.sqrt(len(scalars))) ** 2
        self.assertAlmostEqual(aligned_gain, net["selected_coherent_equal_power_gain"], places=15)
        direct_gain = abs(np.sum(scalars) / np.sqrt(len(scalars))) ** 2
        self.assertAlmostEqual(direct_gain, net["direct_superposition_gain"], places=15)

    def test_ru_common_phase_offset_changes_direct_sum_not_per_ru_power(self):
        baseline = simulate_position(self.cfg["rus"], self.cfg["ue_position_m"], self.codebook)
        shifted_rus = [dict(ru) for ru in self.cfg["rus"]]
        shifted_rus[1]["tx_phase_offset_deg"] = 90.0
        shifted = simulate_position(shifted_rus, self.cfg["ue_position_m"], self.codebook)
        for before, after in zip(baseline["per_ru"], shifted["per_ru"]):
            self.assertAlmostEqual(before["selected_gain"], after["selected_gain"], places=15)
        self.assertNotAlmostEqual(
            baseline["network"]["direct_superposition_gain"],
            shifted["network"]["direct_superposition_gain"],
            places=10,
        )

    def test_svg_plot_contains_both_comparisons(self):
        result = simulate_position(self.cfg["rus"], self.cfg["ue_position_m"], self.codebook)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.svg"
            write_svg(output, result)
            svg = output.read_text(encoding="utf-8")
        self.assertIn("Per-RU effective channel gain", svg)
        self.assertIn("Single-antenna UE: over-the-air coherent sum", svg)
        self.assertIn("Predicted-CSI power vs MRT", svg)
        self.assertTrue(svg.startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
