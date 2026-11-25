# -*- coding: utf-8 -*-
"""
[GC FR3 1117] FR3 첫 점검. 측정 스크립트가 로봇 없이 돌고 있을 때 쓴다:

    # 터미널 1 — 정지 측정 (CSI 를 대시보드 5005 와 이 도구 5009 로 같이)
    NO_ROBOT=1 DURATION=600 LIVE_UDP=127.0.0.1:5005,127.0.0.1:5009 NOTE="FR3 check" python runtime/DMIMO_3TX_Live_Rx.py
    # 터미널 2
    python tools/fr3_check.py               # ① 감시: TX 별 레벨 · 위상 기울기(CFO) · TX 간 위상 흔들림, 1 s 마다
    python tools/fr3_check.py --isolate     # ② RU 하나씩만 켜서 3×3 누설표 (파일럿이 제 자리에 오는지 = 스펙트럼 반전 상쇄)
    python tools/fr3_check.py --sweep       # ③ TX BBox 이득을 내리며 |H| 기울기 → 수신 포화 지점 → 권장 tx_backoff_db
    python tools/fr3_check.py --isolate --sweep --sim     # 장비 없이 시험 (tools/sim_rig.py)

판정 기준 [추론]: CFO |f| < 0.5 Hz (UD·USRP 잠김) · 누설 대각 대비 −20 dB 아래 · 스윕 기울기 0.8~1.2 dB/dB 가 선형 구간.
②·③ 은 BBox 를 만지므로 beam_tracker 를 끈 채로 한다 (TLKCore 를 한 프로세스만 잡게).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beam import geometry as geo                       # noqa: E402
from tools.csi_rx import CsiRx, db, phase_rate_hz, wait_for_csi   # noqa: E402
from tmytek.tmy_ctl import TMY, addrs_of, setup_all_bbox, cfg_gain           # noqa: E402

CFO_OK_HZ = 0.5
LEAK_OK_DB = -20.0


def monitor(rx, every):
    print("시간   |H0| |H1| |H2| [dB]   CFO0 CFO1 CFO2 [Hz]   ∠H1/H0 ∠H2/H0 흔들림 [deg]   pkts")
    while True:
        t, h = rx.collect(every)
        if len(t) < 5:
            print("  CSI 없음 — LIVE_UDP 에 이 포트가 들어갔는지 확인")
            continue
        lv = [float(np.mean(db(h[:, k]))) for k in range(3)]
        cfo = [phase_rate_hz(t, h[:, k]) for k in range(3)]
        rel = [np.degrees(np.std(np.angle(h[:, k] * np.conj(h[:, 0]) * np.exp(-1j * np.angle(np.mean(h[:, k] * np.conj(h[:, 0]))))))) for k in (1, 2)]
        flag = " ⚠ CFO — UD/USRP 기준 잠김 확인" if any(abs(c) > CFO_OK_HZ for c in cfo if np.isfinite(c)) else ""
        print(f"{time.strftime('%H:%M:%S')}  {lv[0]:5.1f} {lv[1]:5.1f} {lv[2]:5.1f}   "
              f"{cfo[0]:+5.2f} {cfo[1]:+5.2f} {cfo[2]:+5.2f}   {rel[0]:5.1f} {rel[1]:5.1f}   {len(t)}{flag}")


def isolate(tmy, rx, cfg, sec):
    """BBox 하나씩만 켠다 (RU 안 교정 전에는 A·B 가 서로 지울 수 있어 RU 단위로 켜면 안 된다)."""
    boxes = cfg["tmytek"]["bbox"]
    print("\n[ISOLATE] BBox 하나만 켜고 TX 별 |H|. '자기' = 그 BBox 의 RU 칸 [dB, 첫 BBox 기준], 누설 = 다른 칸 − 자기 칸")
    print("BBox            자기[dB]  H0(sc18) H1(sc38) H2(sc58)   누설 최대")
    res, ref = [], None
    for on in boxes:
        for b in boxes:
            tmy.switch_channels(b["sn"], off=(b["sn"] != on["sn"]))
        _, h = rx.collect(sec, settle=0.3)
        p = np.mean(np.abs(h) ** 2, axis=0)
        k = on["ru"] - 1
        own = 10 * np.log10(p[k] + 1e-30)
        ref = own if ref is None else ref
        rel = 10 * np.log10(p / max(p[k], 1e-30) + 1e-30)
        leak = max(rel[j] for j in range(3) if j != k)
        res.append((on, own - ref, leak))
        print(f"RU{on['ru']} {on['row']:5s} {on['sn'][-8:]:>8s}  {own - ref:+6.1f}   " + "  ".join(f"{v:7.1f}" for v in rel) + f"   {leak:6.1f}")
    for b in boxes:
        tmy.switch_channels(b["sn"], off=False)
    worst = max(r[2] for r in res)
    weak = [f"RU{r[0]['ru']}{r[0]['row'][0]}" for r in res if r[1] < -10]
    if worst > LEAK_OK_DB:
        print(f"⚠ 누설 최대 {worst:.1f} dB — RU↔X310 채널 배선이 바뀌었거나, 수신 UD 를 안 거쳐 스펙트럼 거울상이 됐을 수 있다")
    else:
        print(f"✅ 누설 최대 {worst:.1f} dB — 파일럿이 제 자리 (반전 상쇄 · 배선 맞음)")
    if weak:
        print(f"⚠ 다른 BBox 보다 10 dB 넘게 약함: {weak} — 케이블·스플리터·AA-Kit 연결, 빔 방향 확인")
    return res


def sweep(tmy, rx, cfg, sec, backoffs):
    t = cfg["tmytek"]
    boxes = t["bbox"]
    print("\n[SWEEP] TX BBox 6대 이득 = 최대 − backoff. 선형이면 |H| 가 1 dB/dB 로 따라온다")
    rows = []
    for bo in backoffs:
        for b in boxes:
            dr = tmy.dr.get(b["sn"]) or [0.0, 0.0]
            tmy.set_beam(b["sn"], max(float(dr[0]), float(dr[1]) - bo), 0.0, 0.0, b.get("dphi_deg", 0.0))
        _, h = rx.collect(sec, settle=0.3)
        rows.append((bo, [float(np.mean(db(h[:, k]))) for k in range(3)]))
    print("backoff  |H0| |H1| |H2| [dB]   기울기(가장 센 TX, 이전 대비 dB/dB)")
    knee = None
    for i, (bo, lv) in enumerate(rows):
        if i == 0:
            print(f"{bo:6.1f}   {lv[0]:5.1f} {lv[1]:5.1f} {lv[2]:5.1f}")
            continue
        dbo = rows[i - 1][0] - bo                       # 이득이 늘어난 양
        k = int(np.argmax(rows[0][1]))                  # 가장 센 TX 로 본다 (약한 칸은 잡음에 묻힘)
        sl = (lv[k] - rows[i - 1][1][k]) / dbo if dbo else float("nan")
        mark = ""
        if np.isfinite(sl) and sl < 0.8 and knee is None:
            knee = rows[i - 1][0]
            mark = "  ← 여기서 눌리기 시작"
        print(f"{bo:6.1f}   {lv[0]:5.1f} {lv[1]:5.1f} {lv[2]:5.1f}   {sl:5.2f}{mark}")
    cur = t.get("tx_backoff_db", 0.0)
    if knee is None:
        print(f"✅ 스윕 범위에서 포화 없음. 지금 config tx_backoff_db = {cur}")
    else:
        rec = knee + 6.0
        print(f"⚠ backoff {knee} dB 보다 세게 쏘면 포화 (수신 BBox IP1dB −32 dBm 또는 X310 수신). "
              f"권장 tx_backoff_db ≥ {rec:.0f} (지금 {cur}) — 로봇이 RU 에 가장 가까이 가는 자리에서 다시 확인")
    for b in boxes:                                     # config 값으로 되돌림
        tmy.set_beam(b["sn"], cfg_gain(t, tmy.dr.get(b["sn"]), "tx"), 0.0, 0.0, b.get("dphi_deg", 0.0))
    return rows, knee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--csi-udp", default="0.0.0.0:5009")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--backoffs", type=float, nargs="+", default=[30, 26, 22, 18, 14, 10, 6, 3, 0])
    ap.add_argument("--sec", type=float, default=1.0, help="한 단계 측정 시간")
    ap.add_argument("--sim", action="store_true", help="장비 없이 (tools/sim_rig.py)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = geo.load_config(a.config)

    if a.sim:
        from tools.sim_rig import SimRig
        tmy = rx = SimRig(cfg)
    else:
        rx = CsiRx(a.csi_udp)
        if not wait_for_csi(rx):
            print(f"CSI 가 {a.csi_udp} 로 안 온다. 측정 스크립트를 LIVE_UDP=127.0.0.1:5005,127.0.0.1:5009 로 돌렸는지 확인")
            return
        tmy = None
        if a.isolate or a.sweep:
            tmy = TMY(cfg["tmytek"].get("working_root", "."), dry_run=a.dry_run)
            tmy.open([b["sn"] for b in cfg["tmytek"]["bbox"]], addrs_of(cfg["tmytek"]["bbox"]))
    try:
        if a.isolate or a.sweep:
            if not a.sim:
                setup_all_bbox(tmy, cfg, rx=False)
            if a.isolate:
                isolate(tmy, rx, cfg, a.sec)
            if a.sweep:
                sweep(tmy, rx, cfg, a.sec, a.backoffs)
            return
        monitor(rx, a.every)
    except KeyboardInterrupt:
        pass
    finally:
        if tmy is not None and not a.sim:
            tmy.close()


if __name__ == "__main__":
    main()
