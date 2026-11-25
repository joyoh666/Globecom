# Globecom D-MIMO 3-RU 데모 코드 (2026-11-17)

Summit 판(`../../6_Summit데모_1110/channel-sounder`, 11-10)을 바탕으로 **X310 3대 · 2.2 GHz · 실시간 예측/스케줄링 · 로봇 이동 3모드 · 빔 추적**으로 고쳤다.
UD Box·BBox(TMYTEK, 7.4 GHz)가 붙는 날 더할 것은 맨 아래.

```
code/
├─ runtime/DMIMO_3TX_Live_Rx.py   TX 3 + RX 1 (X310 3대, 한 프로세스) · 5 ms CSI → UDP 5005 · 저장
├─ runtime/motion.py              로봇 이동: scenario(한 경로 계속) / random(무작위 경유점) / manual(사람이 조종)
├─ runtime/hy_robot_proc.py       로봇 자식 프로세스 (ROS) · 위치 → UDP 5006(대시보드) + 5007(빔 추적기)
├─ live_demo/live_dashboard.py    실시간: CSI → GRU 예측(+25~125 ms) → DQN RU 선택 · 위상 · 로봇 위치 · 빔 각도
├─ live_demo/replay_sender.py     장비 없이 리허설 (저장 trial 의 CSI·위치를 실시간으로 다시 보냄)
├─ beam/beam_tracker.py           위치 → RU 별 방위각 → BBox setBeamAngle (10 Hz) · 빔 상태 → UDP 5008
├─ beam/geometry.py               좌표 변환 (이동 시작 자세 기준)
├─ tmytek/tmy_ctl.py              TLKCore 래퍼 (없으면 dry-run)
├─ tmytek/tmytek_setup.py         FR3 날 시작 설정: UD Box 4대 + BBox 6대
├─ config/demo_config.json        RU 위치·방향 · TMYTEK SN · 주파수 · 빔 추적 파라미터 (현장에서 채움)
├─ hj_common.py · dataset_dashboard.py   Summit 판 그대로 (모델 로드·사후 분석). hj_common 은 Windows 에서 .pt 를 읽게 한 줄 고침
└─ models/                        Summit 모델: GRU best_validation_nmse.pt · DQN final_best_model.pt
```

## 배선 (2.2 GHz, X310 3대)
- X310 #1 (`X1_ADDR`, 기본 192.168.60.2): RF A = TX0(RU1) · RF B = TX1(RU2). 클럭 내부
- X310 #2 (`X2_ADDR`, 기본 192.168.70.3): RF A = TX2(RU3)
- X310 #3 (`X3_ADDR`, 기본 192.168.50.2): RF A = RX(UE). ⚠ 실제 IP 로 바꿔 줄 것
- REF/PPS 체인: #1 OUT → #2 IN, #2 OUT → #3 IN (Octoclock 없음)
- 파일럿 부반송파 TX0/1/2 = 18/38/58 (FDM), 1.4 MHz, 1.92 MS/s

## 실행 — 터미널 3개 (모두 `code/` 에서)
```bash
# ① 대시보드
python live_demo/live_dashboard.py
# ② (선택) 빔 추적. TMYTEK 없으면 --dry-run (각도만 계산·표시). 비교용 고정은 --fixed
python -m beam.beam_tracker --dry-run
# ③ 측정 + 로봇 — 아래 셋 중 하나
```

**A. 시나리오 하나를 계속** (경로가 정해져 있어 재현됨. 학습에 안 쓴 번호를 쓴다 — 학습 S66~71·74·75, 시험 72·73)
```bash
MOTION=scenario SCENARIO_ID=72 DURATION=60 python runtime/DMIMO_3TX_Live_Rx.py
```
경유점 목록을 끝까지 가면 처음으로 돌아가 DURATION 초 동안 반복한다.

**B. 랜덤 이동** (반경 0.5 m 안 경유점을 즉석에서 뽑음. 속도·다리 길이·회전은 학습 경로와 같은 범위)
```bash
MOTION=random DURATION=60 python runtime/DMIMO_3TX_Live_Rx.py          # 재현하려면 MOTION_SEED=1234
```

**D. RU 를 바라본 채 이동 (`facing`, 11-18 — FR3 데모 기본)** — 로봇 머리가 늘 RU 셋 가운데 방향 ±25° 안, 전진·후진으로 반경 0.5 m 안을 오간다.
로봇 위 4×4 수신 배열이 RU 를 늘 앞에 둔다. 같은 seed 면 같은 경로.
```bash
MOTION=facing MOTION_SEED=72 DURATION=60 python runtime/DMIMO_3TX_Live_Rx.py      # 폭은 FACE_DEV_DEG (기본 25)
```
방향은 config 의 RU 위치에서 계산한다 (출발 자세 기준) → **로봇을 RU 쪽을 보게 놓고 출발**. ⚠ Create3 후진 안전 제한은 [모름] — 후진이 멈추면 `safety_override` 확인.

**C. 사람이 조종** (로봇은 위치만 보냄, 조종은 로봇 라즈베리파이에서 `scripts/run_turtlebot4_teleop.sh`)
```bash
MOTION=manual DURATION=120 python runtime/DMIMO_3TX_Live_Rx.py
```
정지 점검: `NO_ROBOT=1 DURATION=20 python runtime/DMIMO_3TX_Live_Rx.py`

세 모드 모두 **USRP 를 한 번 열고 DURATION 동안 끊김 없이** 받는다 (Summit 판은 15 s 마다 재시작). 옛 배치 방식은 `BATCH_MODE=1`.
위치·위상(CSI)은 계속 잡히고, 대시보드가 GRU 예측과 DQN 스케줄링을 실시간으로 돌린다.
주요 환경변수: `X1_ADDR X2_ADDR X3_ADDR` · `FC`(기본 2.2e9) · `TX_GAIN` `RX_GAIN` · `NOTE` · `SAVE_ROOT`.

## 장비 없이 시험
```bash
python live_demo/live_dashboard.py --headless 34 &                # 또는 GUI
python -m beam.beam_tracker --dry-run &
python live_demo/replay_sender.py <11-10 Summit trial 폴더들> --restart-each
```
2026-11-17 이 PC 에서 통과 (Summit S01·S02 재생): 패킷 6010/6010 · 재시작 1회 감지 · GRU 2.9 ms/25 ms 스텝 ·
NMSE +25/+125 ms GRU −16.6/−11.0 dB 대 LR −12.9/−0.5 dB · 빔 추적 각도 10 Hz 송출.

⚠ 확인된 것 두 가지
- **스케일은 자동(처음 2 s)으로 맞춰야 한다.** Summit 사후 그림은 학습 스케일 8.71 을 그대로 써서 11-10 데이터(전력이 300배 작음)에서 GRU NMSE 가 +5 dB 로 망가졌다. 자동 스케일(149)이면 −16.6 dB.
- **DQN 은 약하다.** 같은 재생에서 DQN 3.07 대 oracle 4.16 · random 2.82 b/s/Hz, 최선 선택 9 %, 핸드오버 77 %. 학습 데이터가 10-26 시나리오 6개뿐이다.

## FR3 날(UD Box·BBox 설치) — 코드는 준비됨 (11-17 밤, 시나리오 하나 계속 모드만)

> **11-19 변경: UE = TMYTEK 6–30 GHz 지향성 안테나 1개 → UD#3 → X310 #3** (TMYTEK 이 TX BBox 6대만 준비, RX BBox 없음). 빔 전환은 수백 ms (TMYTEK 추정) · 그룹 일괄 broadcast 는 TMYTEK 이 구현 중 · 최악이면 0° 고정(`BEAM=fixed`). config `rx_bbox` 는 비어 있고 UE 빔은 꺼진다. UD 는 3대(UD#1 = RU1+RU2 2채널, TMYTEK 제안) + 예비 1. 아래의 RX BBox·UPA·4×1 서술은 UPA 로 돌아갈 때를 위해 남긴다.

(옛) UE 수신 = 4×4 phased array UPA (BBox Lite FR3, RX 모드). 추가된 것:

| 파일 | 하는 일 |
|---|---|
| `run_fr3_scenario.sh` | 한 번에: UD 잠금 확인 → 대시보드 → 빔 추적 → 측정+로봇 (시나리오 = config `scenario.id`, 기본 72). `BEAM=fixed`·`SETUP=full`·`CHECK=1`·`DRY=1 REPLAY=…` |
| `tools/level_budget.py` | 레벨 어림: X310 → UD → 스플리터 → TX BBox → 공간 → RX BBox(IP1dB −32 dBm) → UD → X310. 시작 이득을 정한다 |
| `tools/fr3_check.py` | 첫 점검: ① 감시(TX 별 레벨·CFO·위상 흔들림) ② `--isolate` BBox 하나씩 → 누설표(파일럿 제자리 = 반전 상쇄·배선) ③ `--sweep` TX 이득 스윕 → 포화 지점 → 권장 `tx_backoff_db` |
| `tools/calib_ru.py` | RU 안 교정: A만·B만 → Δφ, 위상기 부호는 ±Δφ 둘 다 켜서 센 쪽으로 → `--write` 로 config `dphi_deg` |
| `tools/csi_rx.py` · `tools/sim_rig.py` | CSI UDP 수신기 · 장비 없이 시험하는 가짜 장비(`--sim`) |
| `tmytek/tmy_ctl.py` | BBox TX/RX 모드 · 이득 = 최대 − backoff (`cfg_gain`) · `setup_all_bbox` |
| `beam/beam_tracker.py` | TX 6대 + **UE RX BBox 빔** (`geometry.rx.policy`: `fixed` 배열 정면 / `centroid` 로봇 회전을 상쇄해 RU 셋 가운데 쪽). 대시보드 지도에 노란 점선 |
| 측정 스크립트 | `LIVE_UDP` 가 쉼표로 여러 곳 (대시보드 5005 + 도구 5009) |

**FR3 날 순서** (모두 `code/` 에서)
1. config 채우기: UD 3대·TX BBox 6대 SN (RX BBox 없음, 11-19), RU 위치·정면 방향(로봇 출발 자세 기준), `rx.mount_yaw_deg`(로봇 정면 대비 UPA 정면)
2. TLKCore(FR3 판, `pip install tlkcore` 또는 받은 판을 `code/lib/`) → `python -m tmytek.tmytek_setup` (REF·PLO 잠금 확인). 교정·AAKit 파일은 없다 (TMYTEK 11-18: 챔버 측정 못 함)
3. `python tools/level_budget.py --x310-dbm <스펙트럼 분석기로 잰 값>` → 시작 `tx_backoff_db`·`rx_backoff_db` (기본 30 / 10)
4. 터미널 1: `NO_ROBOT=1 DURATION=900 LIVE_UDP=127.0.0.1:5005,127.0.0.1:5009 TX_GAIN=<낮게> RX_GAIN=<낮게> python runtime/DMIMO_3TX_Live_Rx.py`
   터미널 2: `python tools/fr3_check.py` (CFO ≈ 0 Hz 인지) → `--isolate` → `--sweep` (권장 backoff 를 config 에)
   → `python tools/calib_ru.py --write`
5. `phi_sign` 확인: 로봇을 한쪽에 두고 `beam_tracker --fixed-deg` 로 ± 를 넣어 수신이 커지는 쪽 (RU 마다, UE 도)
6. 데모: `bash run_fr3_scenario.sh` (비교 `BEAM=fixed`). 기록 `NOTE` 자동 = "FR3 RF 7.4 GHz · S72 · beam track"
7. (시간 되면) 빔 추적을 켠 채 모은 7.4 GHz 데이터로 GRU·DQN 재학습

장비 없이 시험한 것 (11-17, 이 PC): `fr3_check.py --isolate --sweep --sim` · `calib_ru.py --sim` (숨은 Δφ 를 1° 안으로, 위상기 부호 ± 둘 다) ·
`DRY=1 SETUP=full REPLAY=<11-10 S02> bash run_fr3_scenario.sh` (설정→대시보드→빔 추적 UE 빔 포함→재생 끝까지).
⚠ 실기 미검증: TMYTEK FR3 문서(`../3_TMYTEK_자료/fr3_sample/`, 그쪽도 실기 시험 전 판) 기준이다. UD Box 함수(`setUDFreq` 등)는 그 문서에 없어 옛 판 기준 · `phi_sign` · 레벨 예산의 [추론] 항목.
**BBox 제어 (11-18 FR3 API 로 바꿈, `tmytek/tmy_ctl.py`)**: BBox 한 대 = 그룹 하나(`setGroup`, channel_map 은 config `chmap` 또는 `row`) → `setRFFreq` 7.4 GHz → `setRFMode`.
빔은 `setBeamAngle`(명령 1번), RU 안 B 쪽 dphi 는 그 뒤 `setChannelPhase` 4번. 위상은 배열 기하로만(피치 18.5 mm, 교정표 없음). 이득 0~31 dB (감쇠기) → `tx/rx_backoff_db` 는 31 에서 뺀다.
가짜 tlkcore(문서대로)로 호출 순서·인자·위상 계산을 시험했다 (TMYTEK 11-03 로그의 θ45° 위상과 일치).
✅ **UE 배열 뒤 RU 문제 → `facing` 이동으로 해결 (11-18, 가짜 로봇 시뮬, 120 s)**:

| 이동 | RU 가운데가 UE 정면 ±40° | ±90° | RU 셋 모두 ±60° | 후진 |
|---|---:|---:|---:|---:|
| scenario S72 / S73 | 27 % / 21 % | 47 % / 61 % | 20 % / 19 % | 0 |
| **facing** seed 72 / 1 / 2 | **100 / 100 / 100 %** | 100 % | 98 / 92 / 95 % | 46~47 % |

`run_fr3_scenario.sh` 기본이 facing (config `scenario.motion`). 11-19 UE 가 지향성 안테나 1개라 facing 을 그대로 쓴다 (안테나가 RU 를 등지지 않게). 학습 경로(`MOTION=scenario`)와의 차이는 첫 시험에서 본다. (옛, UPA 일 때) 남는 것: RU 셋을 한 번에 들으려면 UE 는 수평 폭이 넓은 **4×1 세로 서브어레이**가 낫다 (1×4 가로 빔폭 25~30° 로는 ±31° 퍼진 RU 셋을 못 덮음 [추론]) — 케이블 품번으로 정해지므로 TMYTEK 에 요청.
학습 경로는 전진만이라 GRU·DQN 이 후진 섞인 움직임에 그대로 맞는지는 [모름] → 2.2 GHz 시험 때 `MOTION=facing` 으로 같이 본다.

## 11-24 추가 — 빔 각도 입력 GRU · odom 정확도 점검

**① 빔 각도를 GRU 입력에 (승환 제안)** — RU 별 빔 조향각의 [sin, cos] 를 토큰·age 뒤에 붙인다 (`beam/beam_features.py` 한 곳에서 정의)
```bash
python tools/build_beam_dataset.py [--logs <S66~S75 상위 폴더>]   # 10-26 S66~S75 → data_ready_beam/ (기본 경로는 윤태 Dropbox 기준) (X · A 각도 · P 위치, Summit 과 같은 분할·창·scale)
python train_GRU_beam.py                           # 각도 넣은 모델 → checkpoints_beam/gru_angle_hold.pt
python train_GRU_beam.py --no-angle                # 같은 코드·데이터의 기준 모델 (비교용)
python live_demo/live_dashboard.py --predictor-model checkpoints_beam/gru_angle_hold.pt   # 빔 각도는 UDP 5008 (beam_tracker)
```
- 각도 출처: run 시간대를 덮는 `logs/beam_*.csv` 가 있으면 **실제 적용 각도**, 없으면 로봇 위치에서 계산한 '추적기가 넣었을 각도'
- 미래 스텝 각도는 `hold` (t0 값 유지, 대시보드와 같음) · `--future-angle true` 는 미래 각도를 안다고 본 상한
- `hj_common.NumpyPredictor` 는 config `beam_feat_dim` 을 보고 자동으로 각도를 받는다 (없는 옛 모델은 그대로)
- ⚠ 10-26 데이터는 빔이 없고(ISO, 2.2 GHz) RU 자리 기록도 없어 config 의 FR3 자리(가정)로 각도를 계산했다.
  11-24 결과 (seed 123, test S72·S73): 각도 넣음 −20.0 / −6.7 dB 대 넣지 않음 −20.3 / −7.3 dB (+25 / +125 ms) → **이 데이터에선 이득 없음**.
  빔이 실제로 바뀌는 FR3 데이터에서 다시 본다
- 장비 없이 시험 통과 (11-24): numpy ↔ torch 출력 상대오차 4e-7 · S72 재생 + beam_tracker dry-run → 대시보드 빔 패킷 239 · GRU 3.8 ms/스텝

**② odom 정확도 → 빔 각도 오차** (`tools/odom_check.py`, 로봇 쪽 PC 에서 `code/` 기준)
```bash
python -m tools.odom_check manual --points "1,0 1,1 0,1 0,0"   # 바닥 테이프 점으로 옮기며 Enter
python -m tools.odom_check facing --duration 60 --seed 72      # 데모 이동 그대로 뒤 멈춘 자리를 줄자로
python -m tools.odom_check analyze logs/odom_check_*.json      # RU 자리 바꿔 다시 계산 (로봇 없이)
```
좌표는 beam_tracker 와 같은 시작 자세 좌표. 출력 = 점별 위치 오차 cm · 방향 오차 · RU1~3 빔 각도 오차 + 판정(빔 반폭 12.5° 기준) → `logs/odom_check_*.json/csv/_trace.csv`
⚠ 지금 런타임은 dock 을 건너뛴다 (`skip_dock_mode`) — 매 run 의 원점은 출발 자리이고 오차는 run 안에서만 쌓인다

**11-25 추가**: `tools/odom_plot.py <odom_check_*.json>` — 궤적 · 줄자 대 odom · RU 별 빔 각도 오차 그림 (png). odom_check json 에 시작 자세(`start_pose_odom`)를 남긴다.
`tools/build_beam_dataset.py --train <run…> --validation <run…> --test <run…>` — S 번호 없는 새 run (facing) 을 폴더로 직접 나눈다.
승환 할 일 정의 · 코드 묶음: `../5_승환할거_1125/` (`1125_승환_할거.md` · `1125 승환 할 것_code.zip`).
`tools/exp_record.py [--date YYYYmmdd | --blank]` — 실험 기록지 한 장 (RU 위치 · 로봇 위치 오차 요약 · 2.2 GHz 시험 · 각도 GRU · 메모). --date 면 로봇 위치 오차 표를 자동으로 채움
