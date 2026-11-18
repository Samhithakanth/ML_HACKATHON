"""
report/build_report.py
Assembles the single consolidated PDF report from the section markdown files, in the order required by the
Deliverables document (front matter, Sections 2-8). Figures and tables are the ones produced by the experiment
scripts; nothing is computed here.

    python report/build_report.py            ->  report/PathSnap_Report.pdf

Sources (edit these, then rebuild):
    report/REPORT_FRONT_AND_8.md      summary, team table, Section 8
    report/REPORT_SECTION_2.md        Section 2 (Person 1)
    report/REPORT_SECTIONS_3_4_5.md   Sections 3-5 (Persons 2, 3)
    report/REPORT_SECTION_6_AND_7.md  Sections 6-7 (Person 4)
"""
import re
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "PathSnap_Report.pdf"

# ------------------------------------------------------------------ fonts (Unicode: β, σ, ≤, ≈, →, −)
FONT_DIR = Path("C:/Windows/Fonts")
FONTS = {"Body": "segoeui.ttf", "Body-Bold": "segoeuib.ttf", "Body-Italic": "segoeuii.ttf", "Body-BoldItalic": "segoeuiz.ttf",
         "Mono": "consola.ttf"}
if all((FONT_DIR / f).exists() for f in FONTS.values()):
    for name, f in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
    BODY, BOLD, MONO = "Body", "Body-Bold", "Mono"
else:                                     # fallback (non-Windows): DejaVu if present, else Helvetica
    BODY, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

INK, MUTED, ACCENT, RULE, SHADE = (colors.HexColor(c) for c in ("#1f2328", "#57606a", "#1a5fb4", "#d0d7de", "#f3f5f8"))
S = {
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.2, leading=12.6, textColor=INK, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=9.2, leading=12.6, textColor=INK, leftIndent=14, bulletIndent=4,
                             spaceAfter=2),
    "quote": ParagraphStyle("quote", fontName="Body-Italic" if BODY == "Body" else BODY, fontSize=8.6, leading=11.5,
                            textColor=MUTED, leftIndent=8, spaceAfter=6, borderPadding=(4, 4, 4, 6), backColor=SHADE),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=17, leading=21, textColor=ACCENT, spaceBefore=4, spaceAfter=10, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=7.6, leading=9.4, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=7.6, leading=9.4, textColor=INK),
    "caption": ParagraphStyle("caption", fontName=BODY, fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_CENTER,
                              spaceAfter=8),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=6.6, leading=8.2, textColor=INK, backColor=SHADE,
                           borderPadding=4, spaceAfter=8, spaceBefore=2),
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=26, leading=31, textColor=ACCENT, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=12, leading=16, textColor=MUTED, alignment=TA_CENTER),
}
PAGE_W = A4[0] - 3.6 * cm


# ------------------------------------------------------------------ markdown -> flowables
def clean(t):
    t = "".join(ch for ch in t if ord(ch) <= 0xFFFF and ch != "\ufe0f")            # drop emoji
    t = re.sub(r"\$([^$]+)\$", lambda m: m.group(1).replace("\\Delta", "Δ").replace("\\cos", "cos")
               .replace("\\bar{\\phi}", "φ̄").replace("\\mid", "|").replace("\\text", "").replace("{", "").replace("}", ""), t)
    return t


def inline(t):
    t = clean(t)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", lambda m: f'<font face="{MONO}" size="8">{m.group(1)}</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    return t


def table(rows):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    small = n >= 8
    cs = ParagraphStyle("c", parent=S["cell"], fontSize=6.6 if small else 7.6, leading=8.2 if small else 9.4)
    ch = ParagraphStyle("h", parent=S["cellh"], fontSize=6.6 if small else 7.6, leading=8.2 if small else 9.4)
    # column weight: longest cell, capped (long text wraps), with a floor so short numbers never break mid-value
    cap = 26 if n >= 8 else 60
    lens = [max(14 if n >= 8 else 6, max(min(len(clean(r[i])), cap) for r in rows)) + 2 for i in range(n)]
    widths = [PAGE_W * l / sum(lens) for l in lens]
    data = [[Paragraph(inline(c), ch if i == 0 else cs) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SHADE), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 7)]


def image(caption, rel, base):
    p = (base / rel).resolve()
    if not p.exists():
        return [Paragraph(f"[missing figure: {rel}]", S["caption"])]
    img = Image(str(p))
    scale = min(PAGE_W / img.imageWidth, 10.2 * cm / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return [KeepTogether([img, Paragraph(inline(caption), S["caption"])])]


def md_to_flowables(text, base, skip_h1=False):
    out, lines, i = [], text.splitlines(), 0
    para = []

    def flush():
        if para:
            out.append(Paragraph(inline(" ".join(para)), S["body"])); para.clear()

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if st.startswith("```"):
            flush(); j = i + 1; buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(clean(lines[j])); j += 1
            out.append(Preformatted("\n".join(buf), S["code"])); i = j + 1; continue
        if st.startswith("|"):
            flush(); j = i; buf = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                buf.append(lines[j]); j += 1
            out += table(buf); i = j; continue
        m = re.match(r"!\[(.*)\]\((.+)\)\s*$", st)
        if m:
            flush(); out += image(m.group(1), m.group(2), base); i += 1; continue
        m = re.match(r"(#{1,4})\s+(.*)", st)
        if m:
            flush()
            level, txt = len(m.group(1)), m.group(2)
            if level == 1 and skip_h1:
                i += 1; continue
            is_section = bool(re.match(r"\d+\.\s", txt)) or level == 1
            style = "h1" if is_section else ("h2" if level <= 3 and re.match(r"\d+\.\d+\s", txt) else "h3")
            if is_section and any(not isinstance(f, Spacer) for f in out):
                out.append(PageBreak())
            out.append(Paragraph(inline(txt), S[style])); i += 1; continue
        if st == "---":
            flush(); out.append(Spacer(1, 4)); i += 1; continue
        if st.startswith(">"):
            flush(); buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            out.append(Paragraph(inline(" ".join(buf)), S["quote"])); continue
        m = re.match(r"(\s*)([*-]|\d+\.)\s+(.*)", ln)
        if m and (m.group(2) in "*-" or re.match(r"\d+\.", m.group(2))):
            flush()
            indent = len(m.group(1))
            buf = [m.group(3)]; j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r"\s*([*-]|\d+\.)\s+", lines[j]) \
                    and lines[j].startswith(" ") and not lines[j].strip().startswith(("|", "```", "!")):
                buf.append(lines[j].strip()); j += 1
            bullet = "•" if m.group(2) in "*-" else m.group(2)
            sty = ParagraphStyle("b", parent=S["bullet"], leftIndent=14 + indent * 4, bulletIndent=4 + indent * 4)
            out.append(Paragraph(inline(" ".join(buf)), sty, bulletText=bullet)); i = j; continue
        if not st:
            flush(); i += 1; continue
        para.append(st); i += 1
    flush()
    return out


# ------------------------------------------------------------------ document
def title_page():
    return [Spacer(1, 4.5 * cm),
            Paragraph("PathSnap", S["title"]), Spacer(1, 6),
            Paragraph("Distributed Map-Matching and Route Reconstruction", S["subtitle"]), Spacer(1, 20),
            Paragraph("Technical Report", ParagraphStyle("t2", parent=S["subtitle"], fontName=BOLD, textColor=INK)),
            Spacer(1, 8),
            Paragraph("UE24CS352A Machine Learning · PES University · Department of Computer Science and Engineering", S["subtitle"]),
            Spacer(1, 30),
            Paragraph("Team: [team name] &nbsp;·&nbsp; Repository: github.com/Samhithakanth/ML_HACKATHON", S["subtitle"]),
            Spacer(1, 6),
            Paragraph(f"Generated {date.today().isoformat()} by <font face='{MONO}'>report/build_report.py</font>", S["subtitle"]),
            PageBreak()]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, 1.1 * cm, "PathSnap — Technical Report")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE); canvas.line(1.8 * cm, 1.4 * cm, A4[0] - 1.8 * cm, 1.4 * cm)
    canvas.restoreState()


def main():
    read = lambda n: (HERE / n).read_text(encoding="utf-8")
    front, sec8 = read("REPORT_FRONT_AND_8.md").split("\n# 8.", 1)
    story = title_page()
    story += md_to_flowables(front, HERE)
    story += [PageBreak()] + md_to_flowables(read("REPORT_SECTION_2.md"), HERE)
    # Sections 3-5: the file's preamble (partitions, framing, features, protocol, metrics) is shared setup
    s345 = read("REPORT_SECTIONS_3_4_5.md").replace(
        "# PathSnap Technical Report — Sections 3, 4 and 5 (Persons 2 and 3)", "# Experimental setup for Sections 3–5", 1)
    story += [PageBreak()] + md_to_flowables(s345, HERE)
    story += [PageBreak()] + md_to_flowables(read("REPORT_SECTION_6_AND_7.md"), HERE, skip_h1=True)
    story += [PageBreak()] + md_to_flowables("# 8." + sec8, HERE)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.8 * cm, title="PathSnap Technical Report", author="PathSnap team")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
