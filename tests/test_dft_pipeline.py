from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beam.dft_codebook import DFTCodebook
from ml.dft_dataset import FEATURE_NAMES, build_sequences, train_validation_split
from runtime.dft_beam_measurement import geometry_beam_targets


class CodebookTests(unittest.TestCase):
    def test_fr3_codebook_is_dft_and_symmetric(self):
        cb = DFTCodebook(
            num_elements=4,
            oversampling=4,
            rf_hz=7.4e9,
            element_spacing_m=18.5e-3,
            max_abs_angle_deg=40.0,
        )
        self.assertEqual([beam.dft_bin for beam in cb], list(range(-4, 5)))
        self.assertEqual(cb.broadside_index, 4)
        self.assertAlmostEqual(cb[0].angle_deg, -cb[-1].angle_deg, places=7)
        weights = np.asarray(cb.weights(2))
        self.assertAlmostEqual(float(np.sum(np.abs(weights) ** 2)), 1.0, places=6)
        self.assertEqual(cb.nearest_angle(0.1).dft_bin, 0)

    def test_project_config_describes_4x4_tx_and_single_rx(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config" / "demo_config.json").read_text(encoding="utf-8"))
        cb = DFTCodebook.from_config(cfg)
        self.assertEqual(cb.array_geometry["tx_physical_shape"], [4, 4])
        self.assertEqual(cb.array_geometry["tx_active_shape"], [2, 4])
        self.assertEqual(cb.array_geometry["bbox_per_ru"], 2)
        self.assertEqual(cb.array_geometry["channels_per_bbox"], 4)
        self.assertEqual(cb.array_geometry["scan_axis"], "azimuth")
        self.assertEqual(cb.array_geometry["rx_antennas"], 1)

        rus = {int(ru["ru"]): ru for ru in cfg["geometry"]["rus"]}
        pose = {"x": 0.0, "y": 0.0, "vx": 0.1, "vy": 0.0}
        targets, steering = geometry_beam_targets(rus, pose, cb, lead_s=0.25)
        self.assertEqual(targets[2], cb.broadside_index)
        self.assertAlmostEqual(steering[2], 0.0, places=5)

    def test_apply_lead_and_training_horizon_are_independent(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config" / "demo_config.json").read_text(encoding="utf-8"))
        cb = DFTCodebook.from_config(cfg)
        rus = {int(ru["ru"]): ru for ru in cfg["geometry"]["rus"]}
        # RU2 broadside/bin boundary: the short lead keeps broadside while the
        # longer GRU horizon crosses into the neighbouring DFT bin.
        pose = {"x": 0.0, "y": 0.18, "vx": 0.0, "vy": 0.2}
        applied, _ = geometry_beam_targets(rus, pose, cb, lead_s=0.08)
        future_label, _ = geometry_beam_targets(rus, pose, cb, lead_s=0.25)
        self.assertEqual(applied[2], cb.broadside_index)
        self.assertEqual(future_label[2], cb.broadside_index - 1)


class DatasetTests(unittest.TestCase):
    def test_sweep_labels_build_sequences(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            trial.mkdir()
            cb = DFTCodebook(4, 4, 7.4e9, 18.5e-3, 40.0).to_dict()
            (trial / "metadata.json").write_text(json.dumps({"codebook": cb}), encoding="utf-8")

            header = ["host_rx_time", "pose_valid", "phase"]
            for ru in (1, 2, 3):
                header += [f"ru{ru}_{name}" for name in FEATURE_NAMES]
            with open(trial / "dft_measurements.csv", "w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=header)
                writer.writeheader()
                for index in range(80):
                    row = {"host_rx_time": 1000.0 + index * 0.005, "pose_valid": 1, "phase": "hold"}
                    for ru in (1, 2, 3):
                        for feature_no, name in enumerate(FEATURE_NAMES):
                            row[f"ru{ru}_{name}"] = ru + feature_no * 0.1 + index * 0.001
                    writer.writerow(row)

            sweep_header = ["completed_host_time"] + [f"ru{ru}_best_beam" for ru in (1, 2, 3)]
            with open(trial / "dft_sweeps.csv", "w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=sweep_header)
                writer.writeheader()
                writer.writerow({"completed_host_time": 1000.18, "ru1_best_beam": 1, "ru2_best_beam": 2, "ru3_best_beam": 3})
                writer.writerow({"completed_host_time": 1000.36, "ru1_best_beam": 4, "ru2_best_beam": 5, "ru3_best_beam": 6})

            dataset = build_sequences(
                [trial], sequence_length=8, horizon_s=0.02, stride=2,
                max_gap_s=0.01, max_label_delay_s=1.0,
            )
            self.assertGreater(len(dataset), 10)
            self.assertEqual(dataset.x.shape[1:], (8, len(FEATURE_NAMES)))
            self.assertTrue(set(dataset.y.tolist()).issubset({1, 2, 3, 4, 5, 6}))
            train_idx, val_idx = train_validation_split(dataset, 0.2)
            self.assertGreater(train_idx.size, 0)
            self.assertGreater(val_idx.size, 0)
            self.assertLess(dataset.anchor_times[train_idx].max(), dataset.anchor_times[val_idx].min())

    def test_geometry_labels_need_no_sweep_file(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "geometry_trial"
            trial.mkdir()
            cb = DFTCodebook(4, 4, 7.4e9, 18.5e-3, 40.0).to_dict()
            (trial / "metadata.json").write_text(
                json.dumps({
                    "codebook": cb,
                    "label_source": "future_geometry",
                    "geometry_label_lead_s": 0.25,
                }),
                encoding="utf-8",
            )
            header = ["host_rx_time", "pose_valid", "phase"]
            for ru in (1, 2, 3):
                header += [f"ru{ru}_{name}" for name in FEATURE_NAMES]
                header += [f"ru{ru}_geometry_beam"]
            with open(trial / "dft_measurements.csv", "w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=header)
                writer.writeheader()
                for index in range(70):
                    row = {"host_rx_time": 2000.0 + index * 0.005, "pose_valid": 1, "phase": "geometry"}
                    for ru in (1, 2, 3):
                        for feature_no, name in enumerate(FEATURE_NAMES):
                            row[f"ru{ru}_{name}"] = ru + feature_no * 0.1 + index * 0.001
                        row[f"ru{ru}_geometry_beam"] = (ru + index // 20) % 9
                    writer.writerow(row)
            dataset = build_sequences([trial], sequence_length=8, stride=2, max_gap_s=0.01)
            self.assertGreater(len(dataset), 20)
            self.assertEqual(dataset.label_source, "future_geometry")


if __name__ == "__main__":
    unittest.main()
