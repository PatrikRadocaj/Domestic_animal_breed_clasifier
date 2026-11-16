"""Convert EXAM_PREP.md to PDF using reportlab."""

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONTS = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("Arial",        f"{FONTS}/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold",   f"{FONTS}/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{FONTS}/ariali.ttf"))
pdfmetrics.registerFont(TTFont("Courier-U",    f"{FONTS}/cour.ttf"))
pdfmetrics.registerFont(TTFont("Courier-U-Bold", f"{FONTS}/courbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
pdfmetrics.registerFontFamily("Courier-U", normal="Courier-U", bold="Courier-U-Bold")

MD_PATH = Path(r"C:\Users\User\Desktop\Strojno ucenje\EXAM_PREP.md")
PDF_PATH = Path(r"C:\Users\User\Desktop\Strojno ucenje\EXAM_PREP.pdf")

# ── styles ──────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()

MONO = "Courier-U"
SANS = "Arial"

def sty(name, parent=None, **kw):
    base = ss[parent] if parent else ss["Normal"]
    return ParagraphStyle(name, parent=base, **kw)

S = {
    "h1": sty("h1", fontName="Arial-Bold", fontSize=16, spaceAfter=4, spaceBefore=14,
               textColor=colors.HexColor("#111111"), leading=20),
    "h2": sty("h2", fontName="Arial-Bold", fontSize=13, spaceAfter=3, spaceBefore=10,
               textColor=colors.HexColor("#222266"), leading=17),
    "h3": sty("h3", fontName="Arial-Bold", fontSize=11, spaceAfter=2, spaceBefore=7,
               textColor=colors.HexColor("#333333"), leading=14),
    "body": sty("body", fontName=SANS, fontSize=10, leading=14, spaceAfter=3),
    "li": sty("li", fontName=SANS, fontSize=10, leading=13, spaceAfter=1,
              leftIndent=14),
    "code": sty("code", fontName=MONO, fontSize=8.5, leading=11,
                backColor=colors.HexColor("#f0f0f0"), spaceAfter=4, spaceBefore=2),
    "th": sty("th", fontName="Arial-Bold", fontSize=9, leading=12,
              textColor=colors.white),
    "td": sty("td", fontName=SANS, fontSize=9, leading=12),
}

# ── helpers ──────────────────────────────────────────────────────────────────

def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline_fmt(text):
    """Bold **x**, italic *x*, inline code `x`."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", lambda m: f'<font face="Courier" size="9" backColor="#f0f0f0"> {escape(m.group(1))} </font>', text)
    return text

def make_table(rows):
    """rows[0] = header, rows[1:] = data."""
    if not rows:
        return None
    col_count = max(len(r) for r in rows)
    # pad
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    data = []
    for ri, row in enumerate(rows):
        style = S["th"] if ri == 0 else S["td"]
        data.append([Paragraph(inline_fmt(cell.strip()), style) for cell in row])

    col_w = (A4[0] - 4.4*cm) / col_count
    t = Table(data, colWidths=[col_w] * col_count, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#445588")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t

# ── parser ───────────────────────────────────────────────────────────────────

def parse_md(text):
    story = []
    lines = text.splitlines()
    i = 0
    table_rows = []
    code_lines = []
    in_code = False

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        # filter separator rows (---|---)
        clean = [r for r in table_rows if not all(re.match(r"^[-: ]+$", c) for c in r)]
        if clean:
            t = make_table(clean)
            if t:
                story.append(t)
                story.append(Spacer(1, 6))
        table_rows = []

    while i < len(lines):
        line = lines[i]

        # fenced code block
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                block = "\n".join(code_lines)
                story.append(Preformatted(block, S["code"]))
                story.append(Spacer(1, 4))
                code_lines = []
            else:
                flush_table()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # table row
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            i += 1
            continue
        else:
            flush_table()

        # HR
        if re.match(r"^---+$", line.strip()):
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#999999")))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            txt = inline_fmt(m.group(2))
            sname = ["h1", "h1", "h2", "h3"][min(level - 1, 3)]
            p = Paragraph(txt, S[sname])
            if level == 2:
                story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#aaaacc"), spaceAfter=2))
            story.append(p)
            story.append(Spacer(1, 2))
            i += 1
            continue

        # bullet list
        m = re.match(r"^(\s*[-*+])\s+(.*)", line)
        if m:
            indent = len(m.group(1)) - 1
            txt = inline_fmt(m.group(2))
            bul = ParagraphStyle("li_i", parent=S["li"], leftIndent=14 + indent * 12,
                                 bulletIndent=2 + indent * 12)
            story.append(Paragraph(f"• {txt}", bul))
            i += 1
            continue

        # numbered list
        m = re.match(r"^\s*\d+\.\s+(.*)", line)
        if m:
            txt = inline_fmt(m.group(1))
            story.append(Paragraph(f"&nbsp;&nbsp;{txt}", S["li"]))
            i += 1
            continue

        # blank line
        if line.strip() == "":
            story.append(Spacer(1, 4))
            i += 1
            continue

        # normal paragraph
        story.append(Paragraph(inline_fmt(line), S["body"]))
        i += 1

    flush_table()
    return story

# ── build PDF ────────────────────────────────────────────────────────────────

doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

md_text = MD_PATH.read_text(encoding="utf-8")
story = parse_md(md_text)
doc.build(story)
size_kb = PDF_PATH.stat().st_size // 1024
print(f"Done: {PDF_PATH}  ({size_kb} KB)")
