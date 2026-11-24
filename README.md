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

## DFT 빔 + 새 GRU 확장

기존의 **TX0/1/2 = 부반송파 18/38/58 FDM 파일럿과 5 ms CSI 추정은 바꾸지 않았다.** 새 수집기가 CSI UDP와 로봇 UDP를 받아 BBox의 DFT 빔을 제어하고, 두 스트림을 한 행으로 정렬한다. 기존 실측 파일을 복사하지 않고 제어/동기화 계층을 분리했기 때문에 FDM 파형·동기·USRP 저장 형식은 그대로 유지된다.

```
beam/dft_codebook.py              4×4 TX 패널의 4열 azimuth DFT bin ↔ 실제 조향각/위상
runtime/dft_beam_measurement.py   위치 기반 무-sweep DFT 선택·통합 로그·GRU 실시간 빔 적용
runtime/turtlebot_telemetry.py    TurtleBot odom/방향/속도/각속도 → host UDP 5011
ml/dft_dataset.py                 CSI·pose·sweep를 시계열/미래 최적 빔 label로 변환
ml/dft_gru.py                     GRU 분류 모델과 실시간 checkpoint loader
ml/train_dft_gru.py               trial 단위 분리·정규화·학습·검증·checkpoint 저장
run_dft_measurement.sh            위 수집기 + 기존 DMIMO_3TX_Live_Rx.py 동시 실행
```

실제 구조는 **RU마다 물리 4×4 송신 패널 + BBox Lite FR3 두 대**, UE는 **단일 수신 안테나**다. BBox 한 대가 제공하는 독립 위상 채널은 4개이므로 현재 배선은 두 개의 1×4 수평 활성 서브어레이를 하나의 빔으로 결합한다. 따라서 코드북은 4열 azimuth 축에 대해 4배 oversampling한 16-bin DFT 중 ±40° 범위의 `k=-4..4`, 총 9개 빔을 사용한다. 두 BBox에 같은 DFT bin을 적용하고 splitter/BBox 경로 차이는 `dphi_deg`로 교정한다. elevation은 broadside 고정이다.

물리 패널의 16소자를 모두 독립 제어하는 2차원 `(kx, ky)` DFT와는 구분해야 한다. BBox Lite FR3 네 대/RU 또는 동등한 16채널 위상 제어가 있을 때만 full 4×4 2D DFT로 전환할 수 있다. 현재 구조와 모델의 빔 class는 azimuth DFT class다.

### 1. 로봇 텔레메트리

기존처럼 host PC의 `hy_robot_proc.py`가 로봇을 제어한다면 별도 실행이 필요 없다. ROS 2를 TurtleBot Raspberry Pi에서 실행하고 위치만 host로 보낼 때는 로봇에서 다음을 실행한다.

```bash
python3 -m runtime.turtlebot_telemetry --host <HOST_PC_IP>:5011 --rate 20
```

패킷에는 `x/y/yaw`, 시작 자세 기준 `local_x/local_y/local_yaw`, `vx/vy`, 이동 속도·이동 방향·yaw rate와 sequence/time이 들어간다. RF 정렬에는 서로 다른 PC의 wall clock 대신 host의 UDP 수신 시각을 사용한다. 두 PC에는 chrony/NTP도 켜는 것을 권장한다.

### 2. 위치 기반 무-sweep 빔 선택 및 데이터 수집

정상 운용에서는 sweep하지 않는다. host가 받은 UE 위치와 속도로 `geometry_apply_lead_s` 뒤 위치를 외삽하고, 각 RU의 위치·boresight로 조향각을 계산한 다음 가장 가까운 DFT bin을 즉시 적용한다. 기본 적용 lead는 장비/네트워크 지연 보상용 80 ms이고 빔 갱신은 100 ms다. 학습 정답은 이와 별도로 `prediction_horizon_s`(기본 250 ms) 미래 위치에서 계산한다.

host PC가 로봇 제어까지 하는 기본 구성:

```bash
DFT_MODE=geometry DURATION=180 MOTION=facing bash run_dft_measurement.sh
```

로봇에서 위 텔레메트리 송신기와 주행 시나리오를 실행하는 구성. 각 수집 전에 로봇을 RU 좌표를 측정할 때 사용한 출발 위치와 방향에 두고 텔레메트리 송신기를 시작한다. 송신기가 처음 읽은 자세가 좌표계의 원점과 +x 방향이 된다. 호스트는 기존 FDM 파일럿을 송신하면서 각 RU에 선택된 빔 하나를 적용한다.

```bash
ROBOT_SOURCE=external DFT_MODE=geometry DURATION=180 bash run_dft_measurement.sh
```

장비 없이 시험하려면 한 터미널에서 `python3 -m runtime.dft_beam_measurement --mode geometry --dry-run`을 실행하고, 다른 터미널에서 `python3 -m tools.dft_sim_sender --duration 10`을 실행한다. 실측 결과는 `logs/dft_날짜_시간/`에 저장된다.

- `dft_measurements.csv`: 유효한 자세와 안정된 빔으로 수집한 CSI, RU별 거리/방향/이동, 적용 빔, 품질. CSI 입력 주기는 5 ms다.
- `dft_sweeps.csv`: `DFT_MODE=sweep` 교정/평가 때만 sweep별 실측 최적 빔 기록
- `dft_codebook.json`: 빔 class, DFT bin, 각도, 위상 증가량
- `metadata.json`: 입력 schema, packet/drop 수, 실제 dry-run 여부

geometry 모드는 CSI 수신 시점 이전의 최신 자세를 연결한다. 자세가 없거나 `pose_max_age_s`보다 오래된 경우와 빔 전환 후 `geometry_settle_s`(기본 50 ms) 이내에 받은 CSI는 저장에서 제외한다. 전환 대기값은 실장비에서 확인한 BBox와 CSI 전달 지연에 맞춘다. 적용 빔은 `ruN_beam`에, GRU용 미래 위치 기반 라벨은 `ruN_geometry_beam`에 기록한다. LOS를 가정한 수집에서 CSI는 선택된 빔의 수신 품질을 보여준다.

다음 sweep 모드는 온라인 운용이 아니라 좌표/boresight 교정과 geometry 빔 대비 실제 최적 빔 성능평가용으로 남겨 두었다.

```bash
DFT_MODE=sweep DURATION=60 bash run_dft_measurement.sh
```

이때만 `settle_s` 동안 전환 과도구간을 버리고 `measure_s` 중앙값으로 실측 최적 빔을 정한다.

### 3. 새 입력으로 GRU 학습

모델의 RU별 한 시점 입력은 8개다.

1. UE 방향: UE body frame에서 RU 방향의 `sin/cos`
2. UE 이동: RU 쪽 속도, 접선 속도, yaw rate
3. RU–UE 거리
4. 현재 적용 DFT bin(정규화)
5. 현재 빔 품질 `20 log10 |H|`의 EMA

기본 학습 정답은 현재 위치·속도를 `--horizon` 뒤로 외삽해 계산한 geometry DFT 빔이다. 같은 모델을 세 RU에 공유하되 모든 기하/이동을 해당 RU 기준으로 바꿔 넣는다. 수집 메타데이터의 `geometry_label_lead_s`와 학습 `--horizon`은 같아야 하며 다르면 데이터 로더가 오류를 낸다. sweep trial을 넣으면 해당 trial은 실측 최적 빔을 정답으로 사용할 수도 있다.

```bash
python3 -m pip install -r requirements-dft.txt
python3 -m ml.train_dft_gru logs/dft_20261121_120000 logs/dft_20261121_130000 \
  --output models/dft_beam_gru.pt --sequence-length 40 --horizon 0.25
```

trial이 둘 이상이면 validation trial 전체를 분리하고, 하나뿐이면 시간축 뒤쪽을 validation으로 쓴다. checkpoint에는 학습 데이터의 feature mean/std와 DFT 코드북을 함께 저장한다. 실시간 실행 때 코드북이 다르면 적용하지 않고 오류를 낸다.

geometry 계산만으로도 GRU 없이 빔 선택은 가능하다. GRU는 이동 이력, 방향 변화, 현재 빔 품질을 이용해 미래 bin을 예측하고 제어 지연을 흡수하는 경로다. 다만 geometry label만으로 학습하면 반사·차폐 때문에 LOS와 다른 빔이 더 좋은 상황은 배울 수 없다. 그런 channel-optimal 학습에는 일부 sweep/exploration trial을 추가해야 한다.

### 실측 전: 3-RU 4×4 UPA 위치 기반 빔포밍 테스트

`tools/position_beamforming_test.py`는 실제 BBox 연결과 별개로 **RU마다 16소자를 모두 독립 제어할 수 있는 full 4×4 UPA**를 가정한다. 7.4 GHz 단일경로 LOS/Friis 채널에서 UE 위치로 RU별 azimuth/elevation을 구하고, `(u,v)` 공간주파수가 가장 가까운 2-D DFT 빔을 선택한다. 다음 값을 동시에 비교한다.

- 위치로 선택한 DFT 빔과 코드북을 전부 탐색한 oracle 빔
- RU별 완전 CSI 기반 연속위상 MRT
- **예측한 RU별 유효 CSI 위상으로 공통 위상을 보정한 coherent over-the-air sum**
- 위상 보상 없는 direct sum, 가장 강한 단일 RU, perfect-phase DFT 상한, global MRT 상한
- 전체 송신전력을 1로 고정하고 RU마다 1/3씩 할당한 공정한 비교

RU/UE 좌표와 패널 방향은 `config/position_beamforming_test.json`에서 바꾼다. `boresight_azimuth_deg`는 +x에서 반시계 방향, `boresight_elevation_deg`는 위쪽이 +이며 위치 단위는 m다. 기본 DFT oversampling은 4다.
RU별 BBox/LO/케이블 공통 위상을 시험하려면 각 RU의 `tx_phase_offset_deg`를 바꾼다. 이 값은 RU 내부 DFT 빔 모양은 바꾸지 않고 단일 UE 안테나에서의 복소 합만 바꾼다.
유효 CSI 예측 오차의 영향은 RU별 `predicted_csi_phase_error_deg`로 넣는다. 선택 빔 뒤의 예측 유효 채널을 `g_hat_r`라 하면 RU 1을 기준으로 `alpha_r = angle(g_hat_1) - angle(g_hat_r)`를 계산하고, 최종 송신 가중치는 `w_r = f_DFT,r exp(j alpha_r) / sqrt(3)`이다. 즉 배열 내부 DFT 위상은 유지하면서 RU 전체에 공통 위상 하나를 추가한다.

```bash
python3 -m tools.position_beamforming_test
python3 -m tools.position_beamforming_test --ue 0.2 -0.1 0.8 \
  --json logs/position_test.json --csv logs/position_test.csv --plot
python3 -m tools.position_beamforming_test --phase-errors-deg 0 10 -10 --plot
python3 -m unittest tests.test_position_beamforming -v
```

`--plot`은 `plots/position_beamforming_result.svg`에 RU별 빔 이득과 `best RU / direct DFT sum / predicted-CSI coherent / perfect-phase bound / global MRT`를 저장한다. 다른 경로가 필요하면 `--plot <경로.svg>`로 지정한다.

주 결과 `predicted_csi_precoded_gain`은 세 RU가 같은 심볼을 보내되 예측 유효 CSI로 보정한 `|sum(h_r^H f_r exp(j alpha_r))/sqrt(3)|^2`이다. UE는 안테나 1개이고 수신단 결합은 없다. 예측 유효 CSI 복소값을 직접 시험할 때는 `simulate_position(..., predicted_effective_channels=[...])`로 전달한다. `direct_superposition_gain`은 위상 보정 전 기준값이고, `phase-aligned DFT`는 오차 0°인 상한이다. 현재 실장비의 RU당 BBox 2대×4채널 구성은 full 4×4 독립 제어가 아니라 기존 `beam/dft_codebook.py`의 azimuth-only 4소자 모델을 사용해야 한다.

### 4. GRU 실시간 빔 예측/적용

```bash
DFT_MODE=predict DFT_MODEL=models/dft_beam_gru.pt DURATION=120 bash run_dft_measurement.sh
```

40 sample(기본 200 ms) 입력이 쌓이면 100 ms마다 RU 3개의 미래 최적 빔을 동시에 예측하고 바뀐 RU만 BBox에 적용한다. `runtime/dft_beam_measurement.py --min-confidence 0.6`처럼 확률 문턱을 줄 수도 있다. 실측 전에는 반드시 `DRY=1`으로 packet·codebook·checkpoint 호환을 확인한다.

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
로봇 위 단일 수신 안테나가 RU 를 늘 앞에 두도록 한다. 같은 seed 면 같은 경로.
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
