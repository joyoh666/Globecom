#!/usr/bin/env bash
# Sweep-free geometry/GRU DFT selection + existing three-RU FDM measurement.
#
# Collection:
#   DRY=1 DURATION=30 bash run_dft_measurement.sh
#   DFT_MODE=geometry DURATION=180 bash run_dft_measurement.sh
# Calibration/evaluation only:
#   DFT_MODE=sweep DURATION=180 bash run_dft_measurement.sh
#
# Inference:
#   DFT_MODE=predict DFT_MODEL=models/dft_beam_gru.pt bash run_dft_measurement.sh
#
# ROBOT_SOURCE=embedded uses runtime/hy_robot_proc.py on this host.
# ROBOT_SOURCE=external expects runtime/turtlebot_telemetry.py on the robot to
# send to this host's UDP 5011; RF collection then runs with NO_ROBOT=1.
set -euo pipefail
cd "$(dirname "$0")"

CFG=${CONFIG:-config/demo_config.json}
cfgget() { python3 -c "import json;c=json.load(open('$CFG',encoding='utf-8'));print(c$1)"; }

MODE=${DFT_MODE:-geometry}
DUR=${DURATION:-$(cfgget "['scenario']['duration_s']")}
MOT=${MOTION:-$(cfgget "['scenario'].get('motion','facing')")}
SID=${SCENARIO_ID:-$(cfgget "['scenario']['id']")}
SEED=${MOTION_SEED:-$(cfgget "['scenario'].get('seed',72)")}
DEV=${FACE_DEV_DEG:-$(cfgget "['scenario'].get('face_dev_deg',25)")}
RF=$(cfgget "['freq']['rf_ghz']")
ROBOT_SOURCE=${ROBOT_SOURCE:-embedded}
SETUP=${SETUP:-check}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${DFT_OUTPUT:-logs/dft_${STAMP}}
mkdir -p "$OUT"

DRY_ARG=()
if [ "${DRY:-0}" = "1" ]; then DRY_ARG+=(--dry-run); fi
MODEL_ARG=()
if [ "$MODE" = "predict" ]; then
  if [ -z "${DFT_MODEL:-}" ]; then
    echo "DFT_MODE=predict requires DFT_MODEL=<checkpoint>" >&2
    exit 2
  fi
  MODEL_ARG+=(--model "$DFT_MODEL")
fi

CONTROLLER_PID=""
cleanup() {
  if [ -n "$CONTROLLER_PID" ] && kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    kill -TERM "$CONTROLLER_PID" 2>/dev/null || true
    wait "$CONTROLLER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[DFT-RUN] mode=$MODE RF=${RF}GHz duration=${DUR}s robot=$ROBOT_SOURCE setup=$SETUP output=$OUT"
case "$SETUP" in
  full) python3 -m tmytek.tmytek_setup "${DRY_ARG[@]}" ;;
  check) python3 -m tmytek.tmytek_setup --check "${DRY_ARG[@]}" ;;
  skip) ;;
  *) echo "SETUP must be full, check, or skip" >&2; exit 2 ;;
esac

python3 -u -m runtime.dft_beam_measurement \
  --config "$CFG" --mode "$MODE" --csi-udp 0.0.0.0:5010 --pose-udp 0.0.0.0:5011 \
  --output "$OUT" "${DRY_ARG[@]}" "${MODEL_ARG[@]}" \
  > "$OUT/controller.log" 2>&1 &
CONTROLLER_PID=$!
sleep 2
if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
  wait "$CONTROLLER_PID"
fi

NO_ROBOT_VALUE=0
if [ "$ROBOT_SOURCE" = "external" ] || [ "$ROBOT_SOURCE" = "none" ]; then
  NO_ROBOT_VALUE=1
fi

MOTION="$MOT" SCENARIO_ID="$SID" MOTION_SEED="$SEED" FACE_DEV_DEG="$DEV" \
  DURATION="$DUR" NO_ROBOT="$NO_ROBOT_VALUE" ROBOT_LOG_HZ="${ROBOT_LOG_HZ:-20}" \
  LIVE_UDP="127.0.0.1:5010" LIVE_STATUS_UDP="127.0.0.1:5011" \
  NOTE="${NOTE:-FR3 ${RF} GHz · DFT ${MODE} · ${MOT}}" \
  python3 runtime/DMIMO_3TX_Live_Rx.py

cleanup
CONTROLLER_PID=""
echo "[DFT-RUN] complete: $OUT"
