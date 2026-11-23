# -*- coding: utf-8 -*-
"""
[GC 1117] FR3 날(UD Box·BBox 설치) 시작 설정. 측정·데모 전에 한 번 돌린다.

    python -m tmytek.tmytek_setup --dry-run          # 장비 없이 순서만 확인
    python -m tmytek.tmytek_setup                    # UD Box 4대 + TX BBox 6대 + RX BBox 설정, 잠금 확인, 정면 빔
    python -m tmytek.tmytek_setup --check            # 잠금 상태만 다시 본다

순서: ① 스캔·초기화 ② UD#1 마스터(기준 출력) → 나머지 슬레이브(기준 입력) ③ LO 9.6 / RF 7.4 / IF 2.2 GHz
      ④ REF_LOCK·PLO_LOCK 확인 (안 잠기면 T 분기·MCX 케이블 확인) ⑤ TX BBox 6대: 그룹(channel_map)·7.4 GHz·TX
      ⑥ 정면(0°) 빔 + RU 안 dphi ⑦ UE BBox 를 RX 모드·정면 빔
이득 = 31 − backoff (FR3 감쇠기 0~31 dB, config tx_backoff_db / rx_backoff_db). TX 최대면 수신 BBox 가 포화한다.
값은 config/demo_config.json. ⚠ 실기 미검증 — FR3 API 는 TMYTEK 11-18 문서(../3_TMYTEK_자료/fr3_sample) 기준, 교정표 없음.
UD Box 함수는 FR3 문서에 없어 옛 TLKCore 판 기준이다. 첫날 점검 순서대로 확인한다.
"""
from __future__ import annotations

import argparse
import time

from beam import geometry as geo
from tmytek.tmy_ctl import TMY, addrs_of, cfg_gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="설정 없이 UD 잠금만 확인")
    a = ap.parse_args()

    cfg = geo.load_config(a.config)
    f, t = cfg["freq"], cfg["tmytek"]
    udbs, boxes, rxboxes = t["udb"], t["bbox"], t.get("rx_bbox", [])
    assert abs(f["lo_ghz"] - f["if_ghz"] - f["rf_ghz"]) < 1e-6, "RF = LO − IF 가 아니다 (UD Box LO 하한 6 GHz → LO 9.6)"

    tmy = TMY(t.get("working_root", "."), dry_run=a.dry_run)
    tmy.open([u["sn"] for u in udbs] + [b["sn"] for b in boxes + rxboxes], addrs_of(boxes + rxboxes))
    try:
        if not a.check:
            for u in sorted(udbs, key=lambda u: u["role"] != "master"):      # 마스터 먼저
                tmy.setup_udb(u["sn"], u["role"], f["lo_ghz"], f["rf_ghz"], f["if_ghz"], f["bw_mhz"], t.get("ref_khz", 10000))
            time.sleep(2.0)
        bad = []
        for u in udbs:
            st = tmy.udb_locked(u["sn"])
            print(f"[LOCK] {u['name']:10s} {u['sn']}: {st}")
            if not all(bool(v) for k, v in st.items() if k in ("REF_LOCK", "PLO_LOCK")):
                bad.append(u["name"])
        if bad:
            print(f"!!!! 안 잠긴 UD: {bad} — 마스터 출력 → T → MCX 케이블, 기준 주파수(10 MHz) 확인. 이대로면 CFO 가 생긴다.")
        if a.check:
            return
        for b in boxes:
            dr = tmy.setup_bbox(b["sn"], f["rf_ghz"], mode="TX", box=b)
            g = cfg_gain(t, dr, "tx")
            tmy.set_beam(b["sn"], g, 0.0, 0.0, b.get("dphi_deg", 0.0))
            print(f"[BBOX] RU{b['ru']} {b['row']:5s} {b['sn']}: TX · {f['rf_ghz']} GHz · 정면 · 이득 {g} dB (범위 {dr}) · dphi {b.get('dphi_deg', 0.0)}°")
        for b in rxboxes:
            dr = tmy.setup_bbox(b["sn"], f["rf_ghz"], mode="RX", box=b)
            g = cfg_gain(t, dr, "rx")
            tmy.set_beam(b["sn"], g, 0.0, 0.0, b.get("dphi_deg", 0.0))
            print(f"[BBOX] UE  {b['row']:5s} {b['sn']}: RX · {f['rf_ghz']} GHz · 정면 · 이득 {g} dB (범위 {dr})")
        if not rxboxes:
            print("[BBOX] rx_bbox 비어 있음 — UE = 안테나 1개 (11-19), UE 빔 없음. UPA 로 돌아가면 config 에 SN 을 넣는다")
        print("\n[SETUP] 끝. 다음: python tools/fr3_check.py (TX 하나씩 → 파일럿·위상 정지·레벨) → python tools/calib_ru.py")
    finally:
        tmy.close()


if __name__ == "__main__":
    main()
