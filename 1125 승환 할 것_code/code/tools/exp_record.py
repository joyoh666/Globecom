# -*- coding: utf-8 -*-
"""
[GC 1125] 실험 기록지 (워드 한 장).

    python tools/exp_record.py --blank            # 빈 기록지
    python tools/exp_record.py --date 20261125    # 그날 odom_check 결과를 2번 표에 채워서 → logs/20261125_실험기록지.docx
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from beam import geometry as geo  # noqa: E402
from tools.odom_check import evaluate  # noqa: E402


def table(doc, header, rows, widths_cm):
    from docx.shared import Cm
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, header):
        c.text = h
        for r in c.paragraphs[0].runs:
            r.bold = True
    for row in rows:
        for c, v in zip(t.add_row().cells, row):
            c.text = str(v)
    t.autofit = False
    for row in t.rows:
        for c, w in zip(row.cells, widths_cm):
            c.width = Cm(w)
            c.paragraphs[0].paragraph_format.space_after = 0
    return t


def note(doc, text):
    from docx.shared import Pt, RGBColor
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size, r.font.color.rgb = Pt(8.5), RGBColor(0x66, 0x66, 0x66)
    p.paragraph_format.space_after = Pt(1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(CODE_ROOT / "logs"))
    ap.add_argument("--date", default="")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    day = a.date or datetime.now().strftime("%Y%m%d")
    runs = []
    if not a.blank:
        cfg = geo.load_config(a.config)
        for f in sorted(Path(a.logs).glob(f"odom_check_{day}*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            s = evaluate(d["points"], cfg, 12.5)["summary"]
            mode = d.get("mode") or "manual"
            if mode == "facing":
                mode += f" seed {d.get('seed')}"
            runs.append((mode, s["pos_err_cm_max"], s["beam_err_deg_max"], None))
        if not runs:
            print(f"⚠ {a.logs} 에 odom_check_{day}*.json 이 없다 — 2번 표는 빈칸")

    doc = Document()
    from docx.shared import Cm
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)           # A4
        sec.top_margin = sec.bottom_margin = Cm(1.3)
        sec.left_margin = sec.right_margin = Cm(2.2)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "맑은 고딕", Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    st.paragraph_format.space_after = Pt(2)
    h2 = doc.styles["Heading 2"]
    h2.paragraph_format.space_before, h2.paragraph_format.space_after = Pt(8), Pt(2)
    h2.font.size = Pt(12)
    h1 = doc.styles["Heading 1"]
    h1.paragraph_format.space_before, h1.paragraph_format.space_after = Pt(0), Pt(4)

    doc.add_heading("D-MIMO 실험 기록지", level=1)
    doc.add_paragraph(f"날짜: {day[:4]}.{day[4:6]}.{day[6:]}        실험자:")

    doc.add_heading("1. RU 위치 (대략, m)", level=2)
    note(doc, "줄자로 대략: 로봇 출발 자리 = (0, 0), 로봇 정면 = +x, 왼쪽 = +y")
    table(doc, ("", "x", "y"), [("RU1", "", ""), ("RU2", "", ""), ("RU3", "", "")], (3, 4, 4))

    doc.add_heading("2. 로봇 위치 오차", level=2)
    note(doc, "재는 법: 로봇을 줄자로 잰 점에 두고, 로봇이 계산한 위치(odom)와 줄자 위치의 차이 = 위치 오차, "
              "그 차이 때문에 RU 에서 본 빔 방향이 틀어지는 각 = 빔 각도 오차 (odom_check 가 계산)")
    note(doc, "적는 값: odom_check 끝나면 화면 마지막 줄  '최대 위치 오차 ○ cm … 최대 빔 각도 오차 ○°'  의 두 숫자")
    rows = [(m, p, b) for m, p, b, _ in runs] or [("manual (점 4개)", "", ""), ("facing seed 72", "", ""),
                                                ("facing seed 1", "", ""), ("facing seed 2", "", "")]
    table(doc, ("run", "최대 위치 오차 (cm)", "최대 빔 각도 오차 (°)"), rows, (5, 5.5, 5.5))

    doc.add_heading("3. 2.2 GHz 실기 시험", level=2)
    note(doc, "MOTION=facing DURATION=60 python runtime/DMIMO_3TX_Live_Rx.py  한 번 돌리고")
    note(doc, "결과 칸: IP 는 숫자 (예 192.168.60.2), O/X 칸은 O 또는 X, 폴더 · 에러는 화면 글자 그대로 복사")
    table(doc, ("항목", "결과"), [
        ("X310 #1 IP (RU1·RU2 송신)", ""), ("X310 #2 IP (RU3 송신)", ""), ("X310 #3 IP (로봇 수신)", ""),
        ("대시보드에 CSI 들어옴 (O/X)", ""), ("로봇이 움직임 (O/X)", ""),
        ("저장 폴더 이름 (logs/날짜/…)", ""), ("에러 (있으면 마지막 줄 그대로)", "")], (6.5, 9.5))

    doc.add_heading("4. 각도 GRU", level=2)
    note(doc, "학습 끝나면 마지막 줄  '[TEST] … +25ms ○ dB · … · +125ms ○ dB'  의 두 숫자")
    table(doc, ("돌린 명령", "+25 ms (dB)", "+125 ms (dB)"), [
        ("python train_GRU_beam.py", "", ""), ("python train_GRU_beam.py --no-angle", "", "")], (8, 4, 4))

    doc.add_heading("5. 메모 · 질문", level=2)
    t = table(doc, ("",), [("",)], (16.5,))
    t.rows[0]._tr.getparent().remove(t.rows[0]._tr)     # 머리줄 없이 칸 하나
    t.rows[0].height = Cm(1.6)

    out = Path(a.out) if a.out else Path(a.logs) / f"{day}_실험기록지{'_빈칸' if a.blank else ''}.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"[SAVE] {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    main()
