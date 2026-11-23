#!/usr/bin/env bash
# [GC FR3 1117·1118] FR3 데모 한 번에 (7.4 GHz, UD Box + BBox). code/ 에서:
# 이동은 config scenario.motion — 기본 facing (로봇이 RU 쪽을 바라본 채 전진·후진, seed 로 재현).
#
#   bash run_fr3_scenario.sh                         # 잠금 확인 → 대시보드 → 빔 추적 → 측정+로봇
#   MOTION=scenario SCENARIO_ID=73 bash run_fr3_scenario.sh   # 옛 75개 경로 (로봇이 사방을 봄 → UE 배열 뒤에 RU)
#   BEAM=fixed bash run_fr3_scenario.sh              # 비교용: 빔 정면 고정 (track | fixed | off)
#   SETUP=full bash run_fr3_scenario.sh              # UD·BBox 를 처음부터 설정 (full | check | skip)
#   CHECK=1 bash run_fr3_scenario.sh                 # fr3_check 감시를 같이 (logs/fr3_check_*.txt)
#   DRY=1 REPLAY="<11-10 trial 폴더들>" bash run_fr3_scenario.sh   # 장비 없이 리허설 (TMYTEK dry-run + 저장 데이터 재생)
#
# 먼저 끝나 있어야 하는 것: tools/fr3_check.py --isolate --sweep (레벨) → tools/calib_ru.py --write (RU 안 Δφ).
# 끝내기: Ctrl+C — 뒤에 띄운 대시보드·빔 추적기도 같이 내린다.
set -euo pipefail
cd "$(dirname "$0")"
CFG=config/demo_config.json
cfgget() { python -c "import json,sys;c=json.load(open('$CFG',encoding='utf-8'));print(c$1)"; }

MOT=${MOTION:-$(cfgget "['scenario'].get('motion','facing')")}
SID=${SCENARIO_ID:-$(cfgget "['scenario']['id']")}
SEED=${MOTION_SEED:-$(cfgget "['scenario'].get('seed',72)")}
DEV=${FACE_DEV_DEG:-$(cfgget "['scenario'].get('face_dev_deg',25)")}
if [ "$MOT" = "scenario" ]; then TAG="S${SID}"; else TAG="${MOT} seed ${SEED}"; fi
DUR=${DURATION:-$(cfgget "['scenario']['duration_s']")}
RF=$(cfgget "['freq']['rf_ghz']")
BEAM=${BEAM:-track}
SETUP=${SETUP:-check}
DRY_ARG=""; [ "${DRY:-0}" = "1" ] && DRY_ARG="--dry-run"
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "[FR3] 이동 ${TAG} · ${DUR} s · RF ${RF} GHz · 빔 ${BEAM} · 설정 ${SETUP} ${DRY_ARG}"

# ① UD Box 잠금 / 설정
case "$SETUP" in
  full)  python -m tmytek.tmytek_setup $DRY_ARG ;;
  check) python -m tmytek.tmytek_setup --check $DRY_ARG ;;
  skip)  ;;
esac

# ② 대시보드 (CSI 5005 · 위치 5006 · 빔 5008)
python live_demo/live_dashboard.py ${DASH_ARGS:-} > "logs/dash_${STAMP}.txt" 2>&1 & PIDS+=($!)

# ③ 빔 추적 (위치 5007 → TX 6대. UE RX BBox 는 config rx_bbox 가 있을 때만 — 11-19 이후 없음)
case "$BEAM" in
  track) python -m beam.beam_tracker $DRY_ARG > "logs/beam_${STAMP}.txt" 2>&1 & PIDS+=($!) ;;
  fixed) python -m beam.beam_tracker --fixed $DRY_ARG > "logs/beam_${STAMP}.txt" 2>&1 & PIDS+=($!) ;;
  off)   ;;
esac

LIVE="127.0.0.1:5005"
if [ "${CHECK:-0}" = "1" ]; then
  LIVE="127.0.0.1:5005,127.0.0.1:5009"
  python tools/fr3_check.py > "logs/fr3_check_${STAMP}.txt" 2>&1 & PIDS+=($!)
fi
sleep 2

# ④ 측정 + 로봇 (DURATION 동안) — 또는 리허설 재생
if [ -n "${REPLAY:-}" ]; then
  # shellcheck disable=SC2086
  python live_demo/replay_sender.py $REPLAY --restart-each
else
  MOTION="$MOT" SCENARIO_ID="$SID" MOTION_SEED="$SEED" FACE_DEV_DEG="$DEV" DURATION="$DUR" LIVE_UDP="$LIVE" \
  NOTE="${NOTE:-FR3 RF ${RF} GHz · ${TAG} · beam ${BEAM}}" \
  python runtime/DMIMO_3TX_Live_Rx.py
fi
echo "[FR3] 끝. 로그 logs/*_${STAMP}.txt · 빔 logs/beam_*.csv · 측정 SAVE_ROOT(기본 ./logs/날짜)"
