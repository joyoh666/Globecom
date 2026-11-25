# -*- coding: utf-8 -*-
"""
[GC 1124] 측정 로그 → '빔 각도를 입력에 넣는 GRU' 학습 데이터.

    python tools/build_beam_dataset.py                                   # 10-26 S66~S75, config 의 RU 기하
    python tools/build_beam_dataset.py --logs <run 폴더들의 상위> --config <RU 기하 json> --out data_ready_beam
    python tools/build_beam_dataset.py --train logs/20261202/*facing* --validation <폴더> --test <폴더>   # FR3 새 run (S 번호 없음)

출력 (out/{train,validation,test}/X.npz):
    X   (N, L, 3) complex64   CSI × scale  (Summit 학습과 같은 단위)
    A   (N, L, 3) float32     5 ms 틱마다 RU 별 빔 각도 [deg]
    P   (N, L, 2) float32     로봇 위치 (시작 자세 좌표, m) — 기하를 바꿔 각도를 다시 계산할 때 쓴다
    + manifest.csv · out/dataset_manifest.json

각도의 출처 (run 마다 manifest 의 angle_source 열):
    beam_log  run 시간대를 덮는 logs/beam_*.csv 가 있으면 beam_tracker 가 실제로 넣은 각도
    geometry  없으면 로봇 위치에서 '추적기가 넣었을 각도' (빔 없는 2.2 GHz ISO 측정은 늘 이것)

분할·창·scale 기본값은 Summit 학습(6_Summit데모_1110/channel-sounder/data_ready/dmimo_3tx_s66_s75)과 같다
→ 각도만 더한 효과를 같은 조건에서 비교할 수 있다.

⚠ 10-26 측정 때의 RU 위치는 기록이 없다. 기본 config 는 FR3 데모용 자리(가정)다.
   그날 배치를 알면 --config 로 넣고, 모르면 각도 = '가정한 RU 자리에서 본 로봇 방향' 이다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from beam import beam_features as bf  # noqa: E402
from beam import geometry as geo  # noqa: E402

DT = 0.005
DEFAULT_LOGS = CODE_ROOT.parents[1] / "5_실험_1026_3TX_S66-S75"
SUMMIT_SPLIT = {"train": [66, 68, 69, 71, 74, 75], "validation": [67, 70], "test": [72, 73]}
SUMMIT_SCALE = 8.707947923094595


def load_run(d: Path):
    z = np.load(d / "usrp_sparse_h.npz")
    mk = json.loads((d / "scenario_marker.json").read_text(encoding="utf-8"))
    t, c = np.asarray(z["csi_t_usrp"], float), np.asarray(z["csi"])
    t0u, t1u = float(mk["t0_usrp_est"]), float(mk["t1_usrp_est"])
    sel = (t >= t0u) & (t < t1u)
    c, t = c[sel], t[sel]
    if np.isnan(c).any():
        raise ValueError(f"{d.name}: CSI 에 NaN")
    t0w = float(mk["t0_wall"])
    t_wall = t - t0u + t0w                                   # USRP 시각 → 벽시계 (marker 의 t0 대응)

    ev = mk.get("scenario_event", mk)
    sp = ev.get("start_pose") or mk.get("start_pose")
    rows = list(csv.DictReader(open(d / "turtlebot_states.csv", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{d.name}: turtlebot_states.csv 가 비었다")
    pt = np.array([float(r["t_os"]) for r in rows])
    px = np.array([float(r["x"]) for r in rows])
    py = np.array([float(r["y"]) for r in rows])
    if sp is None:
        sp = {"x": px[0], "y": py[0], "yaw_deg": float(rows[0]["z"])}
    x = np.interp(t_wall, pt, px)
    y = np.interp(t_wall, pt, py)
    sx, sy = bf.odom_to_start(x, y, float(sp["x"]), float(sp["y"]), float(sp["yaw_deg"]))
    cover = float(min(pt[-1], t_wall[-1]) - max(pt[0], t_wall[0])) / max(1e-9, t_wall[-1] - t_wall[0])
    return c.astype(np.complex64), t_wall, np.stack([sx, sy], -1).astype(np.float32), cover


def load_beam_logs(beam_dir: Path):
    logs = []
    for f in sorted(beam_dir.glob("beam_*.csv")):
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:                                    # noqa: BLE001
            continue
        if not rows:
            continue
        t = np.array([float(r["t_os"]) for r in rows])
        deg = np.array([[float(r[f"ru{k}_deg"]) if r[f"ru{k}_deg"] not in ("", "None", "nan") else np.nan
                         for k in (1, 2, 3)] for r in rows])
        logs.append((f, t, deg))
    return logs


def beam_from_logs(logs, t_wall):
    """run 시간대를 다 덮는 beam 로그가 있으면 각 틱 직전에 적용된 각도 (계단, (T,3)). 없으면 None."""
    for f, t, deg in logs:
        if t[0] <= t_wall[0] and t[-1] >= t_wall[-1]:
            idx = np.clip(np.searchsorted(t, t_wall, side="right") - 1, 0, len(t) - 1)
            a = deg[idx]
            if np.isfinite(a).all():
                return a.astype(np.float32), f.name
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(DEFAULT_LOGS), help="run 폴더(*_S??_*)들을 찾을 상위 폴더")
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG), help="RU 기하 (geometry.rus · beam.max_deg)")
    ap.add_argument("--beam-log-dir", default=str(CODE_ROOT / "logs"))
    ap.add_argument("--out", default=str(CODE_ROOT / "data_ready_beam"))
    ap.add_argument("--scale", type=float, default=SUMMIT_SCALE, help="0 이면 train 평균 전력으로 새로 (1e-2 목표)")
    ap.add_argument("--window-sec", type=float, default=6.0)
    ap.add_argument("--stride-sec", type=float, default=0.5)
    ap.add_argument("--split", default="", help='JSON, 예 {"train":[66,68],"validation":[67],"test":[72]} (기본 Summit 분할)')
    ap.add_argument("--train", nargs="*", default=None, help="run 폴더를 직접 지정 (facing 등 S 번호 없는 새 run). 주면 --logs/--split 무시")
    ap.add_argument("--validation", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    a = ap.parse_args()

    cfg = geo.load_config(a.config)
    rus, max_deg = cfg["geometry"]["rus"], float(cfg["beam"]["max_deg"])
    split = json.loads(a.split) if a.split else SUMMIT_SPLIT
    sid2split = {int(s): k for k, v in split.items() for s in v}
    beam_logs = load_beam_logs(Path(a.beam_log_dir))

    runs = {}
    if a.train is not None:                                  # [1125] 폴더 직접 지정 — 번호는 0,1,2… 로 붙인다
        cands, sid2split = [], {}
        for part in ("train", "validation", "test"):
            for d in getattr(a, part):
                sid2split[len(cands)] = part
                cands.append((len(cands), Path(d)))
        split = {k: [s for s, v in sid2split.items() if v == k] for k in ("train", "validation", "test")}
    else:
        cands = []
        for d in sorted(Path(a.logs).rglob("*_S[0-9][0-9]_*")):
            if d.is_dir():
                cands.append((int(re.search(r"_S(\d\d)_", d.name).group(1)), d))
    for sid, d in cands:
        if not (d / "usrp_sparse_h.npz").exists() or not (d / "scenario_marker.json").exists():
            print(f"⚠ 건너뜀 (usrp_sparse_h.npz · scenario_marker.json 없음): {d}")
            continue
        if sid not in sid2split:
            continue
        c, t_wall, pos, cover = load_run(d)
        ang, src = beam_from_logs(beam_logs, t_wall)
        if ang is None:
            ang, src = bf.steer_track(rus, pos[:, 0], pos[:, 1], max_deg), "geometry"
        runs[sid] = dict(name=d.name, c=c, ang=ang, pos=pos, src=src, cover=cover)
        print(f"  S{sid:02d} {sid2split[sid]:<10} {c.shape[0]} ticks · 각도 {src} · "
              f"RU1/2/3 범위 {np.ptp(ang[:, 0]):.1f}/{np.ptp(ang[:, 1]):.1f}/{np.ptp(ang[:, 2]):.1f}° · 위치 덮음 {cover:.0%}")
    missing = sorted(set(sid2split) - set(runs))
    if missing:
        print(f"⚠ 못 찾은 시나리오: {missing}")
    if not runs:
        raise SystemExit(f"run 이 하나도 없다 — --logs 에 *_S66_* … 폴더들이 든 상위 폴더를 주거나 --train/--test 로 run 폴더를 직접 준다 (지금 --logs {a.logs})")

    scale = a.scale
    if scale <= 0:
        p = np.mean([np.mean(np.abs(runs[s]["c"]) ** 2) for s in runs if sid2split[s] == "train"])
        scale = float(np.sqrt(1e-2 / p))
    L, stride = int(round(a.window_sec / DT)), int(round(a.stride_sec / DT))
    out = Path(a.out)
    summary = {"scale": scale, "window_ticks": L, "stride_ticks": stride, "split": split, "config": str(a.config),
               "rus": rus, "max_deg": max_deg, "feature": "sin/cos of per-RU steer angle, step mean",
               "runs": {f"S{s:02d}": {"run": r["name"], "angle_source": r["src"]} for s, r in runs.items()}}
    for part in ("train", "validation", "test"):
        X, A, P, rows = [], [], [], []
        for s in sorted(r for r in runs if sid2split[r] == part):
            r = runs[s]
            for i, a0 in enumerate(range(0, r["c"].shape[0] - L + 1, stride)):
                X.append(r["c"][a0:a0 + L] * np.float32(scale))
                A.append(r["ang"][a0:a0 + L])
                P.append(r["pos"][a0:a0 + L])
                rows.append({"sample_id": len(rows), "scenario_id": s, "run": r["name"], "start_tick": a0,
                             "angle_source": r["src"]})
        if not X:
            print(f"  {part}: 없음")
            continue
        (out / part).mkdir(parents=True, exist_ok=True)
        np.savez(out / part / "X.npz", X=np.stack(X).astype(np.complex64), A=np.stack(A).astype(np.float32),
                 P=np.stack(P).astype(np.float32), dt=np.float32(DT), scale=np.float64(scale))
        with open(out / part / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary[part] = len(X)
        print(f"  {part}: {len(X)} 창 → {out / part / 'X.npz'}")
    (out / "dataset_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"scale {scale:.4f} · 기록 {out / 'dataset_manifest.json'}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    main()
