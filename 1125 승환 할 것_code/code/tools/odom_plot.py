# -*- coding: utf-8 -*-
"""
[GC 1125] odom_check 결과를 그림 한 장으로 — 실측 뒤 PC 에서 본다 (로봇 필요 없음).

    python tools/odom_plot.py logs/odom_check_YYYYmmdd_HHMMSS.json            # 같은 이름 .png 저장
    python tools/odom_plot.py logs/odom_check_*.json --config <실제 RU 자리 json> --show

왼쪽: 시작 자세 좌표의 odom 궤적 · 줄자 점(●) · odom 점(×) · 오차 화살표 · RU 자리와 정면 방향
오른쪽: 점마다 RU1~3 빔 각도 오차 [deg] 막대 + 빔 반폭 기준선 (±12.5°, 1/3 선)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from beam import geometry as geo  # noqa: E402
from tools.odom_check import evaluate  # noqa: E402


def plot(json_path, config=str(geo.DEFAULT_CONFIG), half_beam=12.5, show=False) -> Path:
    """odom_check json 하나 → 같은 이름 .png. exp_record 가 이 함수를 부른다."""
    set_korean_font()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class _A:  # 아래 본문이 쓰는 이름 그대로
        pass
    a = _A()
    a.half_beam, a.show = half_beam, show
    jp = Path(json_path)
    d = json.loads(jp.read_text(encoding="utf-8"))
    cfg = geo.load_config(config)
    res = evaluate(d["points"], cfg, a.half_beam)
    rus = cfg["geometry"]["rus"]

    # 궤적: trace 는 odom 원좌표 → 시작 자세 좌표로
    trace = Path(str(jp.with_suffix("")) + "_trace.csv")
    tx, ty = [], []
    if trace.exists():
        rows = list(csv.DictReader(open(trace, encoding="utf-8")))
        if rows:
            st = d.get("start_pose_odom")                      # 저장된 시작 자세 (없으면 trace 첫 값)
            sx, sy, syaw = st if st else (float(rows[0][k]) for k in ("odom_x", "odom_y", "odom_yaw_deg"))
            for r in rows:
                x, y = geo.odom_to_start(float(r["odom_x"]), float(r["odom_y"]), sx, sy, syaw)
                tx.append(x)
                ty.append(y)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    if tx:
        ax.plot(tx, ty, color="0.7", lw=1, label="odom 궤적")
    for r in res["rows"]:
        ax.plot(r["true_x"], r["true_y"], "o", color="k", ms=6)
        ax.plot(r["odom_x"], r["odom_y"], "x", color="tab:red", ms=8, mew=2)
        ax.annotate("", (r["odom_x"], r["odom_y"]), (r["true_x"], r["true_y"]),
                    arrowprops=dict(arrowstyle="->", color="tab:red"))
        ax.text(r["true_x"], r["true_y"] + 0.06, f"{r['name']} {r['pos_err_cm']:.0f} cm", fontsize=8, ha="center")
    cols = ["tab:blue", "tab:orange", "tab:green"]
    for i, ru in enumerate(rus):
        ax.plot(ru["x"], ru["y"], "s", color=cols[i], ms=9)
        b = math.radians(ru["boresight_deg"])
        ax.plot([ru["x"], ru["x"] + 0.5 * math.cos(b)], [ru["y"], ru["y"] + 0.5 * math.sin(b)], color=cols[i])
        ax.text(ru["x"], ru["y"] + 0.15, f"RU{ru['ru']}", color=cols[i], ha="center", fontsize=9)
    ax.plot([], [], "ok", label="줄자")
    ax.plot([], [], "x", color="tab:red", label="odom")
    ax.arrow(0, 0, 0.3, 0, width=0.01, color="k")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m] (출발 방향)")
    ax.set_ylabel("y [m] (왼쪽)")
    ax.set_title("위치: 줄자 대 odom")
    ax.legend(fontsize=8, loc="best")

    names = [r["name"] for r in res["rows"]]
    w = 0.8 / len(rus)
    for i, ru in enumerate(rus):
        vals = [r[f"ru{ru['ru']}_beam_err_deg"] for r in res["rows"]]
        bx.bar([k + (i - 1) * w for k in range(len(names))], vals, w, color=cols[i], label=f"RU{ru['ru']}")
    for v, ls in ((a.half_beam, "-"), (a.half_beam / 3, ":")):
        bx.axhline(v, color="tab:red", ls=ls, lw=1)
        bx.axhline(-v, color="tab:red", ls=ls, lw=1)
    bx.set_xticks(range(len(names)), names)
    bx.set_ylabel("빔 각도 오차 [deg]")
    bx.set_title(f"빔 각도 오차 — {res['summary']['verdict']}", fontsize=10)
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3, axis="y")
    s = res["summary"]
    fig.suptitle(f"odom 점검 · 최대 위치 오차 {s['pos_err_cm_max']} cm · 최대 빔 각도 오차 {s['beam_err_deg_max']}° "
                 f"(반폭 {a.half_beam}°)", fontsize=11)
    fig.tight_layout()
    out = jp.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"[SAVE] {out}")
    if a.show:
        plt.show()
    plt.close(fig)
    return out


def set_korean_font():
    try:
        import matplotlib
        from matplotlib import font_manager
        names = {x.name for x in font_manager.fontManager.ttflist}
        for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
            if f in names:
                matplotlib.rcParams["font.family"] = f
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
    except Exception:                                        # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--half-beam", type=float, default=12.5)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    plot(a.json, a.config, a.half_beam, a.show)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    main()
