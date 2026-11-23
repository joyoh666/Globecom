# -*- coding: utf-8 -*-
"""
[GC FR3 1117] RU 안 교정 — 한 RU 의 BBox 두 대(A 위 줄 · B 아래 줄, 1:2 스플리터로 갈라짐)의 위상차 Δφ 를 재서
config 의 B 쪽 dphi_deg 에 넣는다. 안 하면 A·B 가 정면에서 서로 지울 수 있다 (fr3_check --isolate 로 보면 RU 가 약함).

    # 터미널 1 — 로봇 정지(출발 자리), CSI 를 5009 로도
    NO_ROBOT=1 DURATION=900 LIVE_UDP=127.0.0.1:5005,127.0.0.1:5009 NOTE="FR3 calib" python runtime/DMIMO_3TX_Live_Rx.py
    # 터미널 2 (beam_tracker 는 꺼 둔다)
    python tools/calib_ru.py                 # 재기만
    python tools/calib_ru.py --write         # config 에 저장 (옛 파일은 config/demo_config.json.bak_YYYYmmdd_HHMMSS)
    python tools/calib_ru.py --sim           # 장비 없이 (tools/sim_rig.py 의 숨은 정답을 맞히는지)

방법 (RU k 마다, 다른 RU 하나를 기준으로 켜 둔다 → 캡처마다 바뀌는 공통 위상·레벨이 지워진다):
  ① A 만 켬 → r_A = ⟨H_k · H_ref*⟩ / ⟨|H_ref|²⟩      ② B 만 켬 → r_B
  ③ Δφ0 = ∠r_A − ∠r_B.  위상기 부호 규약·스펙트럼 반전 때문에 B 에 넣을 값이 +Δφ0 인지 −Δφ0 인지 [모름]
  ④ 둘 다 켜고 B 에 0 · +Δφ0 · −Δφ0 를 차례로 넣어 |H_k/H_ref| 가 가장 큰 쪽을 고른다
     이상적 결맞음 이득 = (|r_A|+|r_B|)² / (|r_A|²+|r_B|²) — 최대 +3 dB. 고른 값이 그 근처면 성공
정면(0°) 빔에서 잰다. 빔을 틀어도 두 줄 사이 위상차는 (수평 조향이면) 그대로라고 본다 [추론].
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beam import geometry as geo                       # noqa: E402
from tools.csi_rx import CsiRx, wait_for_csi           # noqa: E402
from tmytek.tmy_ctl import TMY, addrs_of, setup_all_bbox         # noqa: E402


def ratio(rx, sec, k, ref):
    _, h = rx.collect(sec, settle=0.3)
    if len(h) == 0:
        raise RuntimeError("CSI 가 안 온다")
    return complex(np.mean(h[:, k] * np.conj(h[:, ref])) / max(np.mean(np.abs(h[:, ref]) ** 2), 1e-30))


def calib_ru(tmy, rx, cfg, ru, sec, gains):
    boxes = cfg["tmytek"]["bbox"]
    mine = [b for b in boxes if b["ru"] == ru]
    a = next(b for b in mine if b["row"] == "upper")
    bb = next(b for b in mine if b["row"] == "lower")
    ref_ru = ru % 3 + 1
    k, ref = ru - 1, ref_ru - 1

    def only(on_sns):
        for b in boxes:
            if b["ru"] == ru:
                tmy.switch_channels(b["sn"], off=(b["sn"] not in on_sns))
            elif b["ru"] == ref_ru:
                tmy.switch_channels(b["sn"], off=(b["row"] != "upper"))      # 기준 RU 는 A 하나만 (자기 A·B 상쇄 피함)
            else:
                tmy.switch_channels(b["sn"], off=True)

    def set_b(dphi):
        tmy.set_beam(bb["sn"], gains[bb["sn"]], 0.0, 0.0, dphi)

    set_b(0.0)
    only({a["sn"]})
    r_a = ratio(rx, sec, k, ref)
    only({bb["sn"]})
    r_b = ratio(rx, sec, k, ref)
    d0 = float(np.degrees(np.angle(r_a) - np.angle(r_b)))
    d0 = (d0 + 180) % 360 - 180
    only({a["sn"], bb["sn"]})
    cand = {}
    for d in (0.0, d0, -d0):
        set_b(d)
        cand[d] = abs(ratio(rx, sec, k, ref))
    best = max((d0, -d0), key=lambda d: cand[d])
    ideal = (abs(r_a) + abs(r_b)) ** 2 / (abs(r_a) ** 2 + abs(r_b) ** 2)
    got = cand[best] ** 2 / ((abs(r_a) ** 2 + abs(r_b) ** 2))
    for b in boxes:
        tmy.switch_channels(b["sn"], off=False)
    res = {"ru": ru, "ref_ru": ref_ru, "sn_b": bb["sn"], "amp_a": abs(r_a), "amp_b": abs(r_b),
           "dphi0_deg": d0, "dphi_deg": round(best, 1), "sign": "+" if best == d0 else "−",
           "gain_vs_0_db": 20 * np.log10(cand[best] / max(cand[0.0], 1e-30)),
           "coh_gain_db": 10 * np.log10(max(got, 1e-30)), "ideal_db": 10 * np.log10(ideal)}
    print(f"RU{ru} (기준 RU{ref_ru}): |A| {abs(r_a):.3f} |B| {abs(r_b):.3f} · Δφ0 {d0:+.1f}° → B 에 {best:+.1f}° ({res['sign']}) · "
          f"0° 대비 {res['gain_vs_0_db']:+.1f} dB · 결맞음 {res['coh_gain_db']:+.2f} dB (이상 {res['ideal_db']:+.2f})")
    if res["coh_gain_db"] < res["ideal_db"] - 1.0:
        print(f"   ⚠ 이상보다 1 dB 넘게 모자람 — 로봇·사람이 움직였거나 위상기 2.8° 계단 외의 문제. 다시 재 본다")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--csi-udp", default="0.0.0.0:5009")
    ap.add_argument("--sec", type=float, default=2.0)
    ap.add_argument("--ru", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--write", action="store_true", help="config 의 B 쪽 dphi_deg 를 고친다")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = geo.load_config(a.config)

    if a.sim:
        from tools.sim_rig import SimRig
        tmy = rx = SimRig(cfg, seed=7)
        gains = {b["sn"]: 49.0 - cfg["tmytek"].get("tx_backoff_db", 0.0) for b in cfg["tmytek"]["bbox"]}
        for b in cfg["tmytek"]["bbox"]:
            tmy.set_beam(b["sn"], gains[b["sn"]], 0, 0, 0.0)
    else:
        rx = CsiRx(a.csi_udp)
        if not wait_for_csi(rx):
            print(f"CSI 가 {a.csi_udp} 로 안 온다. 측정 스크립트를 LIVE_UDP=127.0.0.1:5005,127.0.0.1:5009 로 돌렸는지 확인")
            return
        tmy = TMY(cfg["tmytek"].get("working_root", "."), dry_run=a.dry_run)
        tmy.open([b["sn"] for b in cfg["tmytek"]["bbox"]], addrs_of(cfg["tmytek"]["bbox"]))
        gains = setup_all_bbox(tmy, cfg, rx=False)
    try:
        results = [calib_ru(tmy, rx, cfg, ru, a.sec, gains) for ru in a.ru]
        if a.sim:
            print("\n[SIM] 숨은 정답 (부호 −1 이므로 B 에 넣어야 할 값 = +off):")
            for r in results:
                print(f"   RU{r['ru']}: 정답 {tmy.true_off[r['sn_b']]:+.1f}° · 잰 값 {r['dphi_deg']:+.1f}°")
        if a.sim:
            return
        out = geo.CODE_ROOT / "logs" / f"calib_ru_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n기록 {out}")
        if a.write:
            cpath = Path(a.config)
            shutil.copy(cpath, cpath.with_name(cpath.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"))
            raw = json.loads(cpath.read_text(encoding="utf-8"))
            for r in results:
                for b in raw["tmytek"]["bbox"]:
                    if b["sn"] == r["sn_b"]:
                        b["dphi_deg"] = r["dphi_deg"]
            cpath.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"config 저장: {cpath} (B 쪽 dphi_deg). beam_tracker·tmytek_setup 를 다시 돌리면 반영된다")
    finally:
        if not a.sim:
            tmy.close()


if __name__ == "__main__":
    main()
