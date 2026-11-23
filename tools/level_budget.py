# -*- coding: utf-8 -*-
"""
[GC FR3 1117] 7.4 GHz 레벨 예산 — 첫날 이득을 어디서 시작할지 정하는 어림 계산. 정답은 fr3_check --sweep 이 준다.

    python tools/level_budget.py                              # 기본값
    python tools/level_budget.py --x310-dbm -8 --tx-backoff 25 --dist 1.2 2.5 3.5

데이터시트 [원문]: BBox Lite FR3 BLE-1240-G V0.1.0 — TX 이득 49 dB · 최대 입력 −16.5 dBm · EIRP 37.5 dBm ·
  RX 이득 34 dB · RX IP1dB −32 dBm (ANT 단자) · NF 4.5 dB.  UDB-0630 V1.0.4 — 변환손실 10 dB · TX 출력 P1dB 0 dBm.
[추론] 으로 둔 것: EIRP 가 입력·이득에 선형(데이터시트 한 점에 맞춤) · 소자 이득 5 dBi · RU 안 두 BBox 결맞음 +6 dB
  (RU 안 교정 후) · 수신 4채널 결맞음 +6 dB · 세 RU 합 전력이 같이 들어감.
"""
from __future__ import annotations

import argparse
import math

BBOX_TX_GAIN, BBOX_PIN_MAX, BBOX_EIRP_MAX = 49.0, -16.5, 37.5
BBOX_RX_GAIN, BBOX_RX_IP1 = 34.0, -32.0
UD_LOSS = 10.0
K_FIT = BBOX_EIRP_MAX - (BBOX_PIN_MAX + BBOX_TX_GAIN)          # 데이터시트 한 점에 맞춘 안테나 몫 [추론]


def fspl_db(d_m, f_hz):
    return 20 * math.log10(4 * math.pi * d_m * f_hz / 3e8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-ghz", type=float, default=7.4)
    ap.add_argument("--x310-dbm", type=float, default=-10.0, help="X310 TX 출력 (2.2 GHz IF). 첫날 스펙트럼 분석기로 잰다")
    ap.add_argument("--split-db", type=float, default=3.5, help="1:2 스플리터")
    ap.add_argument("--cable-db", type=float, default=1.0, help="UD→스플리터→BBox 케이블 (7.4 GHz)")
    ap.add_argument("--tx-backoff", type=float, default=20.0, help="TX BBox 이득 = 49 − backoff")
    ap.add_argument("--boxes-per-ru", type=int, default=2)
    ap.add_argument("--dist", type=float, nargs="+", default=[1.5, 2.5, 3.5], help="RU↔로봇 거리 [m]")
    ap.add_argument("--rx-elem-dbi", type=float, default=5.0)
    ap.add_argument("--rx-backoff", type=float, default=0.0)
    ap.add_argument("--rx-cable-db", type=float, default=2.0, help="RX BBox → RX UD (7.4 GHz)")
    ap.add_argument("--margin", type=float, default=6.0, help="RX IP1dB 아래로 둘 여유")
    ap.add_argument("--x310-in-max", type=float, default=-20.0, help="X310 RX 입력 상한으로 둘 값 [추론]")
    a = ap.parse_args()
    f = a.f_ghz * 1e9

    pin = a.x310_dbm - UD_LOSS - a.split_db - a.cable_db
    print(f"[TX] X310 {a.x310_dbm:+.1f} dBm → UD −{UD_LOSS:.0f} → 스플리터 −{a.split_db} → 케이블 −{a.cable_db} ⇒ BBox 입력 {pin:+.1f} dBm "
          f"(최대 {BBOX_PIN_MAX}) {'✅' if pin <= BBOX_PIN_MAX else '⚠ 넘음 — X310 TX_GAIN 을 내린다'}")
    g_tx = BBOX_TX_GAIN - a.tx_backoff
    eirp1 = min(BBOX_EIRP_MAX, pin + g_tx + K_FIT)
    coh = 20 * math.log10(a.boxes_per_ru) if a.boxes_per_ru > 1 else 0.0
    eirp = eirp1 + coh
    print(f"[TX] BBox 이득 {g_tx:.1f} dB (backoff {a.tx_backoff}) ⇒ EIRP 한 대 {eirp1:+.1f} dBm · RU ({a.boxes_per_ru}대 결맞음 +{coh:.0f}) {eirp:+.1f} dBm [추론]")

    print(f"\n[RX] 거리별 (소자 {a.rx_elem_dbi} dBi, 세 RU 같은 세기면 합 +{10*math.log10(3):.1f} dB)")
    print("거리[m]  FSPL   RU 하나 ANT[dBm]  세 RU 합   IP1dB 여유   X310 입력[dBm]")
    worst = -1e9
    worst_out = -1e9
    for d in sorted(a.dist):
        L = fspl_db(d, f)
        p1 = eirp - L + a.rx_elem_dbi
        p3 = p1 + 10 * math.log10(3)
        worst = max(worst, p3)
        out = p3 + BBOX_RX_GAIN - a.rx_backoff + 6.0 - a.rx_cable_db - UD_LOSS     # 4채널 결맞음 +6 [추론]
        worst_out = max(worst_out, out)
        m = BBOX_RX_IP1 - p3
        print(f"{d:6.2f}  {L:5.1f}   {p1:+8.1f}        {p3:+7.1f}    {m:+6.1f} {'✅' if m >= a.margin else '⚠'}      {out:+6.1f}")
    need = max(0.0, worst - (BBOX_RX_IP1 - a.margin))
    print(f"\n⇒ 가장 가까운 거리에서 IP1dB −{a.margin:.0f} dB 를 지키려면 TX backoff 를 {need:.1f} dB 더 → "
          f"tx_backoff_db ≈ {a.tx_backoff + need:.0f} (config) [추론]. 첫날 fr3_check --sweep 으로 확정")
    out_after = worst_out - need
    rxb = max(0.0, out_after - a.x310_in_max)
    print(f"   그 뒤 X310 입력 최대 ≈ {out_after:+.1f} dBm → {a.x310_in_max:.0f} dBm 아래로 두려면 rx_backoff_db ≈ {a.rx_backoff + rxb:.0f} [추론]")
    print("   X310 수신은 RX_GAIN 을 낮게 시작해 측정 스크립트 끝의 'peak |x|' 가 0.1~0.5 가 되게 (1.0 근처면 포화).")


if __name__ == "__main__":
    main()
