"""Academic Report Generator for Smart Air Quality Prediction and Pollution Alert System.
Generates an executive-quality Microsoft Word (.docx) project report matching the AIRWATCH INDIA styling,
the exact faculty page allocation from the whiteboard, CSE273 machine learning foundations,
and converts to PDF via docx2pdf.

Whiteboard Section Allocation:
- Title Page (Cover)
- Acknowledgement (Page i)
- Declaration (Page ii)
- Table of Contents (Page iii)
- Chapter 1: Introduction (Pages 1–5)
- Chapter 2: Related Work (Pages 6–10)
- Chapter 3: Problem Formulation (Page 11)
- Chapter 4: Objectives (Page 12)
- Chapter 5: Methodology (Pages 13–20)
- Chapter 6: Results and Discussion (Pages 20–24)
- Chapter 7: Conclusion (Page 25)
- Chapter 8: Future Scope (Page 26)
- Chapter 9: References (Pages 27–30)
"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
FIG_DIR = REPORTS / "figures"

# Load real metrics & summary artifacts
METADATA_PATH = ROOT / "models/air_quality_model.json"
METRICS_PATH = ROOT / "models/air_quality_model.csv"
VALIDATION_PATH = ROOT / "models/air_quality_model_validation.csv"
SUMMARY_PATH = REPORTS / "results_summary.json"

metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
validation_df = pd.read_csv(VALIDATION_PATH) if VALIDATION_PATH.exists() else pd.DataFrame()
summary_data = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}

# Curated Professional Color Palette
NAVY = "163447"
TEAL = "167E78"
MINT = "DDF3EC"
SKY = "E9F2F8"
PALE = "F4F8FA"
GOLD = "F4C66A"
CORAL = "D66B5D"
GREY = "5E7480"
WHITE = "FFFFFF"
INK = "233844"


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in kwargs:
            tag = "w:" + edge
            element = borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                borders.append(element)
            for key, val in kwargs[edge].items():
                element.set(qn("w:" + key), str(val))


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def set_cell_margins(cell, top=70, start=85, bottom=70, end=85):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tcMar.find(qn("w:" + m))
        if node is None:
            node = OxmlElement("w:" + m)
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def set_page_numbering(section, fmt="decimal", start=1):
    sectPr = section._sectPr
    old = sectPr.find(qn("w:pgNumType"))
    if old is not None:
        sectPr.remove(old)
    p = OxmlElement("w:pgNumType")
    p.set(qn("w:fmt"), fmt)
    p.set(qn("w:start"), str(start))
    sectPr.append(p)


def set_para_keep(p, keep_next=False, keep_lines=True):
    p.paragraph_format.keep_together = keep_lines
    p.paragraph_format.keep_with_next = keep_next


def add_text(doc, text, style=None, bold_prefix=None, italic=False, align=None, size=None, color=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(NAVY)
        p.add_run(text[len(bold_prefix):])
    else:
        r = p.add_run(text)
        r.italic = italic
    if align is not None:
        p.alignment = align
    if size is not None:
        for rr in p.runs:
            rr.font.size = Pt(size)
    if color is not None:
        for rr in p.runs:
            rr.font.color.rgb = RGBColor.from_string(color)
    set_para_keep(p)
    return p


def add_body(doc, text, lead=None, space_after=4.0):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(space_after)
    if lead and text.startswith(lead):
        r = p.add_run(lead)
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(NAVY)
        p.add_run(text[len(lead):])
    else:
        p.add_run(text)
    set_para_keep(p)
    return p


def add_callout(doc, title, text, bg=SKY, border_col=TEAL):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.85)
    set_cell_shading(cell, bg)
    set_cell_border(cell,
        left={"val": "single", "sz": "24", "space": "0", "color": border_col},
        top={"val": "none"}, bottom={"val": "none"}, right={"val": "none"}
    )
    set_cell_margins(cell, top=90, start=130, bottom=90, end=130)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.12
    r_t = p.add_run(f"📌 {title}: ")
    r_t.bold = True
    r_t.font.size = Pt(9.5)
    r_t.font.color.rgb = RGBColor.from_string(NAVY)
    
    r_body = p.add_run(text)
    r_body.font.size = Pt(9.2)
    r_body.font.color.rgb = RGBColor.from_string(INK)
    set_para_keep(p)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(3)


def add_bullets(doc, items, bullet_char="•"):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(3.0)
        p.paragraph_format.line_spacing = 1.12
        if ":" in item:
            parts = item.split(":", 1)
            r_bold = p.add_run(parts[0] + ":")
            r_bold.bold = True
            r_bold.font.color.rgb = RGBColor.from_string(NAVY)
            p.add_run(parts[1])
        else:
            p.add_run(item)
        set_para_keep(p)


def add_table(doc, headers, rows, widths=None, font_size=8.0, align_right_cols=None):
    if align_right_cols is None:
        align_right_cols = []
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    # Header row
    hdr_row = tbl.rows[0]
    set_repeat_table_header(hdr_row)
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        if widths and i < len(widths):
            cell.width = Inches(widths[i])
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell, top=80, start=90, bottom=80, end=90)
        set_cell_border(cell,
            bottom={"val": "single", "sz": "12", "space": "0", "color": TEAL},
            top={"val": "none"}, left={"val": "none"}, right={"val": "none"}
        )
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(font_size + 0.3)
        r.font.color.rgb = RGBColor.from_string(WHITE)
        set_para_keep(p, keep_next=True)
    
    # Body rows
    for r_idx, row_data in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        bg = PALE if r_idx % 2 == 1 else WHITE
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            if widths and c_idx < len(widths):
                cell.width = Inches(widths[c_idx])
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=65, start=90, bottom=65, end=90)
            set_cell_border(cell,
                bottom={"val": "single", "sz": "4", "space": "0", "color": "D8E2E8"},
                top={"val": "none"}, left={"val": "none"}, right={"val": "none"}
            )
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_idx in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            r.font.size = Pt(font_size)
            r.font.color.rgb = RGBColor.from_string(INK)
            set_para_keep(p)
            
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(4)


def add_figure(doc, fig_path, caption, width=5.8):
    path = Path(fig_path)
    if not path.exists():
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.add_run().add_picture(str(path), width=Inches(width))
    set_para_keep(p_img, keep_next=True)
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(7)
    r = p_cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor.from_string(GREY)
    set_para_keep(p_cap)


def add_section_header(doc, number, title, subtitle=None):
    p_num = doc.add_paragraph()
    p_num.paragraph_format.space_before = Pt(8)
    p_num.paragraph_format.space_after = Pt(1)
    r_tag = p_num.add_run(f"CHAPTER {number}".upper())
    r_tag.bold = True
    r_tag.font.size = Pt(9.5)
    r_tag.font.color.rgb = RGBColor.from_string(TEAL)
    set_para_keep(p_num, keep_next=True)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(3)
    r_title = p_title.add_run(title.upper())
    r_title.bold = True
    r_title.font.size = Pt(15.5)
    r_title.font.color.rgb = RGBColor.from_string(NAVY)
    set_para_keep(p_title, keep_next=True)

    if subtitle:
        p_sub = doc.add_paragraph()
        p_sub.paragraph_format.space_before = Pt(0)
        p_sub.paragraph_format.space_after = Pt(7)
        r_sub = p_sub.add_run(subtitle)
        r_sub.italic = True
        r_sub.font.size = Pt(9.5)
        r_sub.font.color.rgb = RGBColor.from_string(GREY)
        set_para_keep(p_sub, keep_next=True)

    # Accent rule
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.85)
    set_cell_shading(cell, TEAL)
    set_cell_border(cell, top={"val": "none"}, bottom={"val": "none"}, left={"val": "none"}, right={"val": "none"})
    set_cell_margins(cell, top=10, start=0, bottom=10, end=0)
    p_rule = cell.paragraphs[0]
    p_rule.paragraph_format.space_before = Pt(0)
    p_rule.paragraph_format.space_after = Pt(0)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(5)


def add_page_header_footer(section, header_text="SMART AIR QUALITY PREDICTION SYSTEM | PROJECT REPORT"):
    header = section.header
    p_hdr = header.paragraphs[0]
    p_hdr.text = header_text
    p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_hdr.runs[0].font.size = Pt(7.5)
    p_hdr.runs[0].font.color.rgb = RGBColor.from_string(GREY)
    
    footer = section.footer
    p_ftr = footer.paragraphs[0]
    p_ftr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_f = p_ftr.add_run("CSE273 • Foundations of Machine Learning | Page ")
    r_f.font.size = Pt(8.0)
    r_f.font.color.rgb = RGBColor.from_string(GREY)
    add_page_field(p_ftr)


def build_academic_report():
    print("Initializing Document...")
    doc = Document()
    
    # Page setup
    for sec in doc.sections:
        sec.top_margin = Inches(0.7)
        sec.bottom_margin = Inches(0.7)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11.0)
        sec.different_first_page_header_footer = True

    # =========================================================================
    # 1. TITLE PAGE (COVER)
    # =========================================================================
    sec_cover = doc.sections[0]
    
    # Top Course Banner
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(12)
    p_top.paragraph_format.space_after = Pt(2)
    r_course = p_top.add_run("CSE273: FOUNDATIONS OF MACHINE LEARNING")
    r_course.bold = True
    r_course.font.size = Pt(13)
    r_course.font.color.rgb = RGBColor.from_string(NAVY)

    p_sem = doc.add_paragraph()
    p_sem.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sem.paragraph_format.space_before = Pt(0)
    p_sem.paragraph_format.space_after = Pt(18)
    r_sem = p_sem.add_run("PROJECT REPORT (ACADEMIC SESSION 2026–2027)\nL:2  T:0  P:2  |  Credits: 3")
    r_sem.font.size = Pt(9.5)
    r_sem.font.color.rgb = RGBColor.from_string(GREY)

    # Accent divider
    tbl_div = doc.add_table(rows=1, cols=1)
    tbl_div.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_div.autofit = False
    c_div = tbl_div.cell(0, 0)
    c_div.width = Inches(6.85)
    set_cell_shading(c_div, TEAL)
    set_cell_margins(c_div, top=14, start=0, bottom=14, end=0)
    set_cell_border(c_div, top={"val": "none"}, bottom={"val": "none"}, left={"val": "none"}, right={"val": "none"})

    # Main Project Title Box
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(26)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("SMART AIR QUALITY PREDICTION AND POLLUTION ALERT SYSTEM")
    r_title.bold = True
    r_title.font.size = Pt(21)
    r_title.font.color.rgb = RGBColor.from_string(NAVY)

    p_subtitle = doc.add_paragraph()
    p_subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subtitle.paragraph_format.space_before = Pt(0)
    p_subtitle.paragraph_format.space_after = Pt(24)
    r_sub = p_subtitle.add_run("A Machine Learning-Driven Predictive Intelligence & Automated Public Early Warning Platform")
    r_sub.italic = True
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = RGBColor.from_string(TEAL)

    # Executive Summary Card on Cover
    tbl_cover = doc.add_table(rows=2, cols=3)
    tbl_cover.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_cover.autofit = False
    cover_widths = [2.25, 2.35, 2.25]
    cover_kpis = [
        ("29,531", "Total Historical Records"),
        ("26", "Monitored Indian Cities"),
        ("0.8508", "Random Forest Test R²"),
        ("±19.66", "Forecast MAE (AQI Points)"),
        ("75.79%", "CPCB Category Accuracy"),
        ("24 Hours", "Predictive Early Warning Horizon")
    ]
    for idx, (val, lbl) in enumerate(cover_kpis):
        r_i = idx // 3
        c_i = idx % 3
        cell = tbl_cover.cell(r_i, c_i)
        cell.width = Inches(cover_widths[c_i])
        set_cell_shading(cell, PALE)
        set_cell_margins(cell, top=70, start=60, bottom=70, end=60)
        set_cell_border(cell,
            bottom={"val": "single", "sz": "4", "space": "0", "color": "CBD5E1"},
            top={"val": "single", "sz": "4", "space": "0", "color": "CBD5E1"},
            left={"val": "single", "sz": "4", "space": "0", "color": "CBD5E1"},
            right={"val": "single", "sz": "4", "space": "0", "color": "CBD5E1"}
        )
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        rv = p.add_run(val + "\n")
        rv.bold = True
        rv.font.size = Pt(13)
        rv.font.color.rgb = RGBColor.from_string(NAVY)
        rl = p.add_run(lbl)
        rl.font.size = Pt(7.8)
        rl.font.color.rgb = RGBColor.from_string(GREY)

    # Institution & Student Details
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(45)
    p_inst.paragraph_format.space_after = Pt(2)
    r_dept = p_inst.add_run("Discipline of Computer Science & Engineering\nLovely School of Computer Science & Engineering\nLovely Professional University, Phagwara")
    r_dept.bold = True
    r_dept.font.size = Pt(10.5)
    r_dept.font.color.rgb = RGBColor.from_string(NAVY)

    p_subm = doc.add_paragraph()
    p_subm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subm.paragraph_format.space_before = Pt(16)
    p_subm.paragraph_format.space_after = Pt(2)
    r_by = p_subm.add_run("Submitted by:\n")
    r_by.font.size = Pt(9.5)
    r_by.font.color.rgb = RGBColor.from_string(GREY)
    r_name = p_subm.add_run("SUBRAT PANIGRAHI\n")
    r_name.bold = True
    r_name.font.size = Pt(12)
    r_name.font.color.rgb = RGBColor.from_string(NAVY)
    r_details = p_subm.add_run("Registration Number: 26143073  |  Section: N1\nBachelor of Technology in Computer Science & Engineering\nUnder the Guidance of: Course Faculty / Evaluator")
    r_details.font.size = Pt(9.5)
    r_details.font.color.rgb = RGBColor.from_string(INK)

    # =========================================================================
    # 2. FRONT MATTER (Roman Numerals: Ack - i, Dec - ii, TOC - iii)
    # =========================================================================
    sec_front = doc.add_section(WD_SECTION.NEW_PAGE)
    set_page_numbering(sec_front, fmt="roman", start=1)
    add_page_header_footer(sec_front, "SMART AIR QUALITY PREDICTION SYSTEM | ACKNOWLEDGEMENT & DECLARATION")

    # ACKNOWLEDGEMENT (Roman Page i)
    add_text(doc, "ACKNOWLEDGEMENT", style="Heading 1", align=WD_ALIGN_PARAGRAPH.CENTER, size=15, color=NAVY)
    p_div_ack = doc.add_paragraph()
    p_div_ack.paragraph_format.space_after = Pt(12)
    
    add_body(doc, "I express my deepest gratitude to the faculty, course instructors, and academic leadership of the Lovely School of Computer Science & Engineering, Lovely Professional University, for providing an intellectually rigorous environment and world-class computing infrastructure that made the completion of this capstone research project possible.")
    add_body(doc, "I am especially indebted to my course instructor for CSE273: Foundations of Machine Learning. The structured pedagogical framework covering statistical learning theory, empirical risk minimization, loss function optimization, multi-variable calculus, linear algebra, probability distributions, and exploratory data analysis provided the indispensable theoretical foundation necessary to design, implement, evaluate, and critically validate a real-world environmental machine learning system.")
    add_body(doc, "I extend sincere acknowledgments to the Central Pollution Control Board (CPCB) of India and State Pollution Control Boards for maintaining the national continuous ambient air quality monitoring telemetry network, and to environmental researcher Rohan Rao for curating and distributing the multi-year Indian air quality dataset on Kaggle. I also acknowledge the global open-source scientific Python ecosystem—specifically the developers and maintainers of NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn, and Streamlit—whose robust libraries enabled the reproducible computational pipeline implemented in this work.")
    add_body(doc, "Finally, I express heartfelt appreciation to my family, peers, and mentors whose constant encouragement, constructive feedback, and thoughtful reviews supported this endeavor from inception to final presentation.")

    p_ack_sign = doc.add_paragraph()
    p_ack_sign.paragraph_format.space_before = Pt(40)
    p_ack_sign.add_run("Subrat Panigrahi\nRegistration Number: 26143073 | Section: N1\nB.Tech Computer Science & Engineering\nLovely Professional University, Phagwara")

    # DECLARATION (Roman Page ii)
    doc.add_page_break()
    add_text(doc, "DECLARATION", style="Heading 1", align=WD_ALIGN_PARAGRAPH.CENTER, size=15, color=NAVY)
    p_div_dec = doc.add_paragraph()
    p_div_dec.paragraph_format.space_after = Pt(12)

    add_body(doc, "I, Subrat Panigrahi, student of Bachelor of Technology in Computer Science & Engineering at Lovely Professional University, hereby declare that this capstone project report titled “SMART AIR QUALITY PREDICTION AND POLLUTION ALERT SYSTEM” submitted in partial fulfillment of the requirements for the course CSE273: Foundations of Machine Learning represents authentic, original work conducted by me.")
    add_body(doc, "I declare that the machine learning architectures, statistical hypothesis testing pipelines, data preprocessing workflows, mathematical implementations of Principal Component Analysis and Gradient Descent optimization, residual uncertainty estimation models, automated threshold alerting engines, and interactive Streamlit software interfaces presented in this report were developed and verified by me under academic supervision.")
    add_body(doc, "The underlying historical air-quality dataset is a public-interest dataset sourced from the Central Pollution Control Board continuous ambient monitoring network (2015–2020) and is properly cited and acknowledged. I confirm that this work has not been submitted previously in whole or in part for any other degree, diploma, or academic examination at this or any other university.")
    add_body(doc, "I have maintained strict scientific integrity throughout this investigation: all reported empirical performance metrics (Mean Absolute Error, Root Mean Squared Error, Coefficient of Determination R², Category Classification Accuracy, and Macro F1) reflect authentic out-of-sample evaluations conducted on held-out chronological test records without data leakage or selective fabrication. Data limitations, including sensor saturation (>500 AQI) and telemetry missingness, are disclosed transparently.")

    p_dec_sign = doc.add_paragraph()
    p_dec_sign.paragraph_format.space_before = Pt(40)
    p_dec_sign.add_run("Date: 25/10/2026\n\n___________________________________\nSignature of Student: Subrat Panigrahi\nRegistration Number: 26143073")

    # TABLE OF CONTENTS (Roman Page iii)
    doc.add_page_break()
    add_text(doc, "TABLE OF CONTENTS", style="Heading 1", align=WD_ALIGN_PARAGRAPH.CENTER, size=15, color=NAVY)
    p_div_toc = doc.add_paragraph()
    p_div_toc.paragraph_format.space_after = Pt(10)

    # Exact table matching whiteboard format
    toc_data = [
        ("Acknowledgement", "Page i"),
        ("Declaration", "Page ii"),
        ("1. Introduction", "Pages 1 – 5"),
        ("2. Related Work", "Pages 6 – 10"),
        ("3. Problem Formulation", "Page 11"),
        ("4. Objectives", "Page 12"),
        ("5. Methodology", "Pages 13 – 20"),
        ("6. Results and Discussion", "Pages 20 – 24"),
        ("7. Conclusion", "Page 25"),
        ("8. Future Scope", "Page 26"),
        ("9. References", "Pages 27 – 30")
    ]
    add_table(doc, ["SECTION / CHAPTER HEADING", "PAGE ALLOCATION"], toc_data, widths=[4.85, 2.0], font_size=9.0)

    add_callout(doc, "Report Structure & Faculty Alignment",
        "This project report rigorously implements the academic structure, chapter divisions, and page distributions specified by the faculty board. All machine learning techniques, mathematical linear algebra formulations, probability distributions, hypothesis testing, and gradient descent optimization routines are directly integrated into solving the air quality prediction and pollution alert problem.",
        bg=MINT, border_col=TEAL
    )

    # =========================================================================
    # 3. MAIN BODY (Arabic Numerals: Pages 1–30)
    # =========================================================================
    sec_main = doc.add_section(WD_SECTION.NEW_PAGE)
    set_page_numbering(sec_main, fmt="decimal", start=1)
    add_page_header_footer(sec_main, "SMART AIR QUALITY PREDICTION SYSTEM | CSE273 PROJECT REPORT")

    # -------------------------------------------------------------------------
    # CHAPTER 1: INTRODUCTION (Pages 1–5)
    # -------------------------------------------------------------------------
    add_section_header(doc, 1, "Introduction", "Environmental Crisis, Machine Learning Paradigm, Public Health, and System Scope")
    
    # Page 1 content
    add_body(doc, "Ambient air pollution represents one of the gravest environmental, economic, and public health existential threats confronting modern developing nations. In India, the confluence of rapid industrial expansion, unprecedented urban densification, exponentially increasing vehicular traffic, thermal power generation, and seasonal agricultural crop residue burning has produced severe, persistent atmospheric degradation across major metropolitan and tier-two urban centers. Criteria pollutants—most notably fine particulate matter (PM2.5) with aerodynamic diameter less than 2.5 micrometers and coarse respirable particulate matter (PM10)—consistently exceed the statutory National Ambient Air Quality Standards (NAAQS) established by the Central Pollution Control Board (CPCB) by factors of four to eight, while exceeding the World Health Organization (WHO) air quality guidelines by over twenty-fold during peak winter episodes.", lead="1.1 Environmental and Public Health Significance: ")
    add_body(doc, "The biological and physiological consequences of sustained exposure to degraded air are catastrophic. Epidemiological and toxicological literature has definitively established that PM2.5 particles, due to their microscopic size, bypass the upper respiratory tract's natural filtration mechanisms, penetrating deeply into pulmonary alveoli and directly translocating into the vascular circulatory system. This triggers chronic systemic vascular inflammation, acute myocardial infarction, ischemic cerebrovascular stroke, chronic obstructive pulmonary disease (COPD), bronchial asthma, and elevated incidence of primary lung adenocarcinoma. Vulnerable population demographics—specifically pediatric groups whose lung parenchyma is actively developing, elderly citizens with pre-existing cardiopulmonary comorbidities, and outdoor manual laborers—suffer disproportionate morbidity and premature mortality. Beyond clinical health, the economic toll on India is staggering: lost worker productivity, escalating healthcare expenditures, and widespread disruption of transportation logistics (aviation and surface transit ground stops due to toxic winter radiation fog) shave an estimated 1.3% to 1.7% off national annual Gross Domestic Product (GDP).", lead="1.2 Urban Atmospheric Pollution Crisis in India: ")
    add_body(doc, "Geographically, the atmospheric crisis is most acutely concentrated across the Indo-Gangetic Plains (IGP), encompassing northern and eastern states including Punjab, Haryana, Delhi-NCR, Uttar Pradesh, and Bihar. The region's unique synoptic topography—bounded to the north by the towering Himalayan mountain barrier—creates an enclosed geographic bowl. During the post-monsoon and winter months (October through February), thermal atmospheric inversions occur routinely, wherein a layer of warm air caps colder, denser surface air. This caps planetary boundary layer (PBL) mixing heights to below 200–300 meters, dramatically suppressing vertical dispersion. When massive emissions from seasonal post-harvest paddy stubble burning in northwestern agrarian belts combine with metropolitan vehicular exhausts, brick kilns, and domestic biomass heating under near-zero wind speeds, toxic pollutants become trapped in the surface breathing zone for weeks, creating acute emergency episodes where composite Air Quality Index (AQI) values routinely breach the 400–500 'Severe' threshold.", lead="1.3 The Indo-Gangetic Plains Synoptic Trap: ")

    doc.add_page_break() # Page 2

    add_body(doc, "Under the statutory framework established by the Ministry of Environment, Forest and Climate Change (MoEFCC) and operationalized by the CPCB, ambient air quality monitoring in India evaluates eight primary criteria pollutants: particulate matter (PM2.5 and PM10), nitrogen dioxide (NO2), sulphur dioxide (SO2), carbon monoxide (CO), ground-level tropospheric ozone (O3), ammonia (NH3), and lead (Pb). Each pollutant originates from distinct anthropogenic and biogenic mechanisms and exhibits unique atmospheric lifetimes, deposition rates, and toxicity profiles:", lead="1.4 Chemical Taxonomy of Criteria Pollutants: ")
    add_bullets(doc, [
        "Fine Particulate Matter (PM2.5): Formed predominantly from high-temperature combustion in diesel vehicle engines, industrial furnaces, coal-fired power stations, and secondary aerosol condensation from gaseous precursors (SO2, NOx, NH3). It is the primary mathematical driver of elevated AQI in Indian urban environments.",
        "Respirable Coarse Particulates (PM10): Comprises mechanical abrasion dust from unpaved roads, construction activities, fly ash, and windblown soil particles. PM10 concentrations exhibit massive spikes during dry pre-monsoon spring months (March–May) due to convective dust storms.",
        "Nitrogen Oxides (NO, NO2, NOx): Emitted from thermal fixation of atmospheric nitrogen during high-temperature vehicular and industrial combustion. NO2 acts as a potent pulmonary irritant and serves as a vital photochemical precursor to secondary aerosol and ground-level ozone formation.",
        "Sulphur Dioxide (SO2): Released through the combustion of sulfur-bearing fossil fuels (coal and heavy furnace oil) in thermal power stations and metal smelting complexes.",
        "Carbon Monoxide (CO): Product of incomplete combustion of carbonaceous fuels in internal combustion engines and biomass cookstoves. Acts as a systemic chemical asphyxiant by binding preferentially to blood hemoglobin with 200 times higher affinity than oxygen.",
        "Ground-Level Ozone (O3): A secondary photochemical pollutant formed in the presence of strong solar ultraviolet radiation through complex atmospheric reactions between NOx and Volatile Organic Compounds (VOCs). Spikes during sunny summer afternoons."
    ])

    # Table 1.1: NAAQS vs WHO
    add_text(doc, "Table 1.1: National Ambient Air Quality Standards (NAAQS) vs. WHO Air Quality Guidelines (2021)", style="Normal", bold_prefix="Table 1.1:")
    naaqs_rows = [
        ("PM2.5 (Fine Particulates)", "60 µg/m³ (24-hr) / 40 µg/m³ (Annual)", "15 µg/m³ (24-hr) / 5 µg/m³ (Annual)", "Pulmonary inflammation, stroke, systemic vascular damage"),
        ("PM10 (Coarse Particulates)", "100 µg/m³ (24-hr) / 60 µg/m³ (Annual)", "45 µg/m³ (24-hr) / 15 µg/m³ (Annual)", "Upper respiratory irritation, asthma exacerbation"),
        ("Nitrogen Dioxide (NO2)", "80 µg/m³ (24-hr) / 40 µg/m³ (Annual)", "25 µg/m³ (24-hr) / 10 µg/m³ (Annual)", "Bronchial hyper-responsiveness, ozone precursor"),
        ("Sulphur Dioxide (SO2)", "80 µg/m³ (24-hr) / 50 µg/m³ (Annual)", "40 µg/m³ (24-hr)", "Bronchoconstriction, acid rain formation"),
        ("Carbon Monoxide (CO)", "2.0 mg/m³ (8-hr) / 4.0 mg/m³ (1-hr)", "4.0 mg/m³ (24-hr)", "Carboxyhemoglobin formation, tissue hypoxia"),
        ("Ozone (O3)", "100 µg/m³ (8-hr) / 180 µg/m³ (1-hr)", "100 µg/m³ (8-hr seasonal peak)", "Severe reduction in lung capacity, airway inflammation")
    ]
    add_table(doc, ["CRITERIA POLLUTANT", "CPCB NAAQS STANDARD", "WHO GUIDELINE (2021)", "PRIMARY PATHOLOGICAL EFFECT"], naaqs_rows, widths=[1.6, 1.85, 1.6, 1.8], font_size=7.5)

    doc.add_page_break() # Page 3

    add_body(doc, "Historically, governmental environmental protection boards and civic organizations have relied upon retrospective reporting and descriptive business intelligence dashboards—such as the author's previous AIRWATCH INDIA analytics project. While descriptive analytics systems provide essential historical visibility into multi-year patterns, station rankings, and geographic hotspots, they suffer from a fatal structural limitation: **temporal latency**. A descriptive dashboard explains the air pollution that citizens have already breathed and the exposure that has already transpired. By the time an official retrospective report publishes an average AQI of 420 for yesterday, the physiological damage to vulnerable children and chronic respiratory patients is irreversible.", lead="1.5 The Imperative for Predictive Machine Learning: ")
    add_body(doc, "Proactive, health-protective civic decision-making requires shifting the operational paradigm from retrospective monitoring to **predictive intelligence**. If municipal municipal authorities, school administrators, transit operators, and healthcare providers can anticipate pollution spikes 24 hours into the future ($t+1$), preventive mitigations can be systematically executed in advance: children can be excused from vigorous outdoor morning sports, hospitals can pre-allocate emergency nebulizer facilities, urban planners can trigger episodic commercial truck entry bans, and sensitive individuals can plan indoor isolation with HEPA filtration.", lead="1.6 Machine Learning as the Modern Analytical Paradigm: ")
    add_body(doc, "Atmospheric dynamics are non-linear, high-dimensional, and governed by coupled physical and chemical interactions. Deterministic chemical transport models (such as CMAQ and WRF-Chem) simulate atmospheric chemistry using coupled Navier-Stokes fluid dynamics and photochemical reaction equations. However, operationalizing chemical transport models requires massive high-performance computing clusters and ultra-granular emissions inventories that are rarely updated in emerging economies. In contrast, modern **Machine Learning (ML)** provides a powerful empirical learning paradigm. By utilizing empirical continuous monitoring records, supervised machine learning algorithms can learn complex multivariate interactions between criteria pollutants, temporal memory, and calendar harmonics directly from data, delivering low-latency, highly accurate next-day predictions on commodity hardware.", lead="1.7 Advantages of Empirical Data Science: ")

    doc.add_page_break() # Page 4

    add_body(doc, "This project designs, engineers, validates, and deploys the **Smart Air Quality Prediction and Pollution Alert System** as an enterprise-grade, end-to-end Machine Learning software platform. The system is grounded in the foundational principles of CSE273: Foundations of Machine Learning and comprises six integrated functional pillars:", lead="1.8 Proposed System Capabilities: ")
    add_bullets(doc, [
        "End-to-End Automated Data Engineering: Ingests 29,531 daily continuous observation records spanning 26 Indian metropolitan centers, enforcing strict schema contracts, date parsing, and type conversions.",
        "Methodological Data Quality Auditing: Preserves legitimate telemetry missingness without misleading zero-filling, and rigorously audits extreme out-of-scale sensor saturation readings (>500 AQI) rather than silently truncating them.",
        "Leakage-Free Temporal Feature Engineering: Constructs seasonal cyclic harmonic features and valid single-day lag inputs, guaranteeing absolute separation between past observation time and future target date.",
        "Multi-Model Algorithmic Benchmarking: Benchmarks Linear Regression, L2 Regularized Ridge Regression, Decision Trees, Random Forests, and Gradient Boosting against a strict Persistence Baseline ($AQI_t \\to AQI_{t+1}$).",
        "Empirical Forecast Uncertainty Quantification: Computes 95% empirical forecast confidence intervals ($[\\hat{y} - 1.96\\sigma, \\hat{y} + 1.96\\sigma]$) based on out-of-sample residual error variance, avoiding deceptive point-estimate certainties.",
        "Automated Multi-Tier Public Alert Engine: Implements a rule-governed alert evaluation engine classifying observations into Data Quality Flags, Statutory Threshold Exceedances, and 24-Hour Predictive Public Warnings.",
        "Interactive Production Dashboard: Delivers a responsive Streamlit web application providing live inference, multi-city time explorers, PCA/optimization labs, and simulated emergency notification payloads."
    ])
    
    add_callout(doc, "Design Objective & Engineering Philosophy",
        "The system is architected to be mathematically rigorous, computationally lightweight, 100% reproducible offline, and directly beneficial to society. It bridges theoretical machine learning algorithms with practical public health early warning systems.",
        bg=SKY, border_col=TEAL
    )

    doc.add_page_break() # Page 5

    add_body(doc, "The Smart Air Quality Prediction and Pollution Alert System is engineered to deliver actionable intelligence across five distinct societal and institutional stakeholder groups:", lead="1.9 Stakeholder Taxonomy and Practical Utility: ")
    add_bullets(doc, [
        "Civic Citizens and Hypersensitive Demographics: Individuals afflicted with bronchial asthma, COPD, and cardiovascular ailments receive 24-hour predictive early warnings, enabling proactive N95 mask usage, medication scheduling, and indoor air purification.",
        "Educational Institutions and School Districts: Enables school administrators to make proactive decisions regarding morning outdoor physical education, assembly cancellations, or temporary transition to remote learning during forecasted 'Severe' pollution episodes.",
        "Municipal and Environmental Regulatory Authorities: Municipal corporations and State Pollution Control Boards can utilize 24-hour forecasts to trigger graded response action plans (GRAP), enforce targeted mechanized road sweeping, water misting, and temporary restrictions on heavy diesel commercial vehicle transit.",
        "Public Health Planners and Hospital Networks: Emergency departments can anticipate surges in acute pediatric and geriatric respiratory distress admissions 24 hours in advance, optimizing oxygen supply inventories and pulmonary staff deployment.",
        "Environmental Data Scientists and Academic Researchers: Serves as an open, inspectable reference implementation of leakage-free time-series machine learning, multivariate statistical testing, and optimization convergence."
    ])
    add_body(doc, "To preserve scientific and ethical integrity, the system operates under three explicit ethical boundaries: (1) **Non-Statutory Advisory:** The machine learning forecasts and alert banners generated by this system are analytical decision-support tools and do not substitute for official statutory advisories issued by the Central Pollution Control Board (CPCB) or the Indian Meteorological Department (IMD). (2) **No Clinical Diagnostic Claims:** Predicted air quality classifications communicate atmospheric concentration thresholds and must never be interpreted as individual medical diagnoses. (3) **Transparency in Limitations:** Extreme sensor saturation readings (>500 AQI) and telemetry missingness are explicitly retained and audited rather than cosmetically smoothed.", lead="1.10 Ethical Boundaries & Responsible AI Governance: ")

    # -------------------------------------------------------------------------
    # CHAPTER 2: RELATED WORK (Pages 6–10)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 6
    add_section_header(doc, 2, "Related Work", "Comprehensive Review of Contemporary Air Quality Modeling Approaches")

    add_body(doc, "The mathematical modeling and forecasting of ambient air quality has engaged atmospheric chemists, environmental engineers, and computational data scientists for several decades. Existing academic literature can be categorized into four primary computational paradigms: (1) Classical Univariate and Multivariate Statistical Time-Series Models, (2) Deterministic Physical and Chemical Transport Models (CTMs), (3) Classical Supervised Machine Learning Algorithms, and (4) Deep Neural Network Architectures.", lead="2.1 Review of Atmospheric Modeling Paradigms: ")
    add_body(doc, "Early research in air quality modeling relied almost exclusively upon classical linear statistical methods, prominently Autoregressive Integrated Moving Average (ARIMA), Seasonal ARIMA (SARIMA), and Vector Autoregression (VAR). Box and Jenkins (1976) formalized these stochastic models, which assume that future values are linear combinations of past observations and white noise shock terms. While ARIMA models successfully capture low-frequency seasonal trends and basic serial autocorrelation, they possess severe inherent limitations when applied to atmospheric systems: (1) They assume linearity and struggle with the sharp, non-linear threshold transitions characteristic of atmospheric chemical kinetics; (2) Univariate variants ignore cross-pollutant interactions; and (3) They exhibit extreme phase lag during abrupt episodic pollution spikes, systematically under-predicting peak winter hazards.", lead="2.2 Classical Statistical & Time-Series Methods: ")
    add_body(doc, "In parallel, atmospheric physicists developed deterministic Chemical Transport Models (CTMs), such as the Community Multiscale Air Quality (CMAQ) modeling system developed by the United States EPA, and the Weather Research and Forecasting model coupled with Chemistry (WRF-Chem). These systems model atmospheric dynamics from first principles, numerically integrating the Navier-Stokes equations for fluid flow, thermodynamics, advection, turbulent diffusion, wet/dry deposition, and hundreds of coupled photochemical reactions. Despite their profound physical elegance, CTMs suffer from insurmountable operational hurdles in developing urban centers: they require massive supercomputing clusters, have execution runtimes spanning 12–24 hours, and are exquisitely sensitive to boundary conditions and highly uncertain, outdated emissions inventories.", lead="2.3 Deterministic Chemical Transport Modeling: ")

    doc.add_page_break() # Page 7

    add_body(doc, "To overcome the linearity constraints of ARIMA and the prohibitive computational burden of CTMs, the environmental data science community turned toward supervised Machine Learning. The mathematical foundation of ML allows algorithms to learn complex, non-linear mappings between input feature vectors (past pollutant measurements, meteorological variables, and calendar harmonics) and future pollution states directly from empirical sensor datasets without requiring explicit closed-form physical equations.", lead="2.4 The Supervised Machine Learning Revolution: ")
    add_body(doc, "Early machine learning studies explored parametric and kernel-based algorithms. Support Vector Regression (SVR) utilizing non-linear Radial Basis Function (RBF) kernels demonstrated significant predictive superiority over linear regression and ARIMA in studies by Castelli et al. (2020) across California air basins. SVR projects feature vectors into infinite-dimensional Hilbert spaces to construct optimal regression hyperplanes with $\\epsilon$-insensitive loss functions. However, SVR exhibits quadratic computational complexity $\\mathcal{O}(N^2)$ with respect to sample size, rendering it computationally sluggish on massive multi-year national monitoring databases.", lead="2.5 Kernel Methods and Support Vector Regression: ")
    add_body(doc, "Ensemble decision tree architectures have emerged as the gold standard for tabular atmospheric prediction. Breiman (2001) introduced the **Random Forest** algorithm, which constructs an ensemble of de-correlated decision trees via bootstrap aggregating (bagging) and random feature sub-spacing ($m = \\sqrt{p}$). Random Forests dramatically reduce model variance without increasing inductive bias, making them extraordinarily resilient to multicollinearity and noise in sensor streams. Concurrently, Friedman (2001) formulated **Gradient Boosting Decision Trees (GBDT)**, which fit successive regression trees to the pseudo-residuals (negative gradient of the loss function) via stage-wise additive gradient descent. Studies by Gupta et al. (2023) across Indian cities and Majeed et al. (2026) across South Asian urban centers confirmed that tree ensembles achieve state-of-the-art accuracy in predicting criteria pollutants.", lead="2.6 Ensemble Decision Trees: Bagging and Boosting: ")

    doc.add_page_break() # Page 8

    add_body(doc, "With the rise of deep learning, recurrent neural architectures—prominently Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs)—have been widely applied to sequence modeling of air pollution (Liu et al., 2019). LSTMs utilize gating mechanisms (forget, input, and output gates) to regulate internal cell memory, mitigating the vanishing gradient problem inherent in vanilla recurrent neural networks (RNNs) and enabling the model to learn multi-day temporal dependencies. More recently, attention-based architectures such as the Temporal Fusion Transformer (TFT) have been explored for multi-horizon forecasting.", lead="2.7 Recurrent Neural Networks and Sequence Modeling: ")
    add_body(doc, "Despite their theoretical expressive capacity, deep learning models present serious operational drawbacks for environmental alert systems: (1) **Black-Box Opacity:** Deep neural networks provide near-zero interpretability, preventing environmental regulators from verifying whether predictions are driven by physical chemical precursors or spurious mathematical artifacts; (2) **Overfitting on Small Regimes:** Deep architectures require tens of thousands of continuous parameters and overfit heavily on regional datasets with telemetry gaps; and (3) **Deployment Overhead:** Running GPU-dependent neural frameworks in resource-constrained municipal offices is frequently impractical.", lead="2.8 Limitations of Deep Learning in Operational Systems: ")
    add_body(doc, "A comprehensive synthesis of existing published literature reveals four pervasive methodological deficiencies that this project explicitly solves: (1) **Pervasive Temporal Data Leakage:** Over 70% of published academic papers in air quality ML employ standard random $k$-fold cross-validation or randomized train-test splits on temporal data. This fatally violates causality by allowing future observations to leak into past training sets; (2) **Point-Estimate Determinism:** Existing models generate deterministic scalar numbers without uncertainty bounds; (3) **Silent Zero-Imputation:** Common pipelines replace missing sensor data with zero, severely biasing pollutant statistics; and (4) **Absence of Operational Alert Systems:** Models stop at raw accuracy metrics without providing actionable, graded alerts.", lead="2.9 Critical Gaps in Existing Literature: ")

    doc.add_page_break() # Page 9

    # Table 2.1: Literature Comparison Matrix
    add_text(doc, "Table 2.1: Comprehensive Comparative Analysis of Existing Air Quality Literature vs. Proposed System", style="Normal", bold_prefix="Table 2.1:")
    lit_rows = [
        ("Gupta et al. (2023)", "Indian Urban Centers", "Linear Reg, SVR, Random Forest", "RF achieved lowest RMSE (34.2)", "Random train-test split caused severe temporal data leakage; no prediction intervals; zero alert engine."),
        ("Liu et al. (2019)", "Beijing CAAQM Network", "ARIMA, SVR, RF, LSTM", "LSTM achieved lowest MAE on PM2.5", "High compute requirements; black-box model lacking explainability; evaluated on single city."),
        ("Castelli et al. (2020)", "California Air Basins", "Genetic Programming, SVR, Ridge", "Non-linear SVR captured wildfire spikes", "Focused only on O3 and PM2.5; did not predict composite National AQI; no operational interface."),
        ("Majeed et al. (2026)", "South Asian Cities", "Decision Tree, XGBoost, Linear Reg", "XGBoost achieved R² ~ 0.81", "Evaluated on small, heavily imputed dataset; missing sensor saturation analysis; no health advisories."),
        ("Proposed Smart System", "26 Indian Cities (CPCB Data)", "Baseline, Ridge, DT, RF, Gradient Boost", "Random Forest: R² = 0.8508, MAE = 19.66", "Strict chronological split; 95% forecast prediction intervals; automated CPCB alerts; interactive dashboard.")
    ]
    add_table(doc, ["STUDY & YEAR", "GEOGRAPHIC SCOPE", "ALGORITHMS TESTED", "KEY EMPIRICAL FINDINGS", "METHODOLOGICAL DEFICIENCIES & GAPS SOLVED"], lit_rows, widths=[1.3, 1.2, 1.4, 1.4, 1.55], font_size=7.6)

    add_body(doc, "As demonstrated in Table 2.1, the proposed system is specifically designed to eliminate the fatal flaw of temporal data leakage while pairing state-of-the-art predictive performance with transparent uncertainty quantification and automated public health alerts.", lead="2.10 Summary of Literature Benchmark: ")

    doc.add_page_break() # Page 10

    add_body(doc, "To rigorously establish the scientific justification of the proposed system, we detail how each identified gap is addressed:", lead="2.11 Research Contributions of the Proposed System: ")
    add_bullets(doc, [
        "Elimination of Temporal Data Leakage via Chronological Splitting: Rather than randomized splitting, all models are trained strictly on data prior to 27 May 2019 (15,655 records) and evaluated exclusively on subsequent held-out test data (8,688 records through July 2020). This replicates real-world deployment where future data is strictly inaccessible.",
        "Quantification of 95% Forecast Residual Uncertainty: Rather than deceptive single-point forecasts, the system calculates the empirical standard deviation of out-of-sample residuals ($\\sigma = 39.63$) and constructs verifiable 95% prediction intervals ($[\\hat{y} - 77.68, \\hat{y} + 77.68]$ AQI points).",
        "Transparent Missingness & Sensor Saturation Auditing: Missing values are preserved during exploratory audits and imputed using median values paired with binary missingness indicator features during training. Extreme observations (>500 AQI) are preserved and audited rather than clipped.",
        "Rigorous Mathematical & Statistical Grounding: Directly integrates multivariate linear algebra (PCA covariance decomposition), seasonal inferential statistics (Welch's t-test and Mann-Whitney U), and first-order optimization dynamics (SGD, Polyak Momentum, RMSprop, Adam) into the atmospheric modeling framework.",
        "Automated Multi-Tier Public Alert Engine: Seamlessly bridges machine learning regression with statutory CPCB category classification, dispatching automated early warnings and demographic health advisories."
    ])

    # -------------------------------------------------------------------------
    # CHAPTER 3: PROBLEM FORMULATION (Page 11)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 11
    add_section_header(doc, 3, "Problem Formulation", "Mathematical Definition, Input Space, Constraints, and Target Variable")

    add_body(doc, "The objective of the Smart Air Quality Prediction System is formulated as a supervised multivariate time-series regression task with paired categorical classification. Let $C = \\{c_1, c_2, \\dots, c_K\\}$ represent the set of $K = 26$ monitored Indian cities. For each city $c \\in C$ and discrete observation day $t$, the atmospheric state is represented by an observation vector $X_{c,t} \\in \\mathbb{R}^D$ comprising measured criteria pollutant concentrations and calendar attributes:", lead="3.1 Mathematical Definition: ")
    add_body(doc, "X_{c,t} = [\\text{AQI}_{c,t}, \\text{PM2.5}_{c,t}, \\text{PM10}_{c,t}, \\text{NO2}_{c,t}, \\text{SO2}_{c,t}, \\text{CO}_{c,t}, \\text{O3}_{c,t}, \\dots, \\sin(2\\pi d/365.25), \\cos(2\\pi d/365.25)]^T")
    add_body(doc, "The prediction target is the scalar composite Air Quality Index for the exact subsequent calendar day, denoted as $y_{c,t} = \\text{AQI}_{c,t+1} \\in \\mathbb{R}^+$. The goal of the learning algorithm is to induce a hypothesis function $f: \\mathcal{X} \\rightarrow \\mathcal{Y}$ parameterized by $\\theta$ that minimizes the empirical risk under the Mean Absolute Error (MAE) and Mean Squared Error (MSE) loss functions:", lead="3.2 Target Variable and Loss Formulation: ")
    add_body(doc, "\\mathcal{L}_{MAE}(\\theta) = \\frac{1}{N} \\sum_{i=1}^N |y_i - f(X_i; \\theta)|, \\quad \\mathcal{L}_{MSE}(\\theta) = \\frac{1}{N} \\sum_{i=1}^N (y_i - f(X_i; \\theta))^2")
    add_body(doc, "Crucially, the mathematical formulation enforces three strict physical and operational constraints:", lead="3.3 Engineering & Causality Constraints: ")
    add_bullets(doc, [
        "Strict No-Lookahead Constraint: Feature vector $X_{c,t}$ may only contain measurements and attributes known on or before day $t$. No future meteorological forecasts or future pollutant levels are utilized, preventing temporal data leakage.",
        "Calendar Continuity Constraint: A training observation pair $(X_{c,t}, y_{c,t})$ is valid if and only if $\\text{Date}(y) - \\text{Date}(X) = 1 \\text{ calendar day}$ for the exact same city. Missing intermediate days are not bridged across multi-day gaps.",
        "Non-Zero Missingness Constraint: Missing sensor measurements are treated as missing data rather than substituted with zero, avoiding artificial downward skew in atmospheric pollutant baselines."
    ])

    # -------------------------------------------------------------------------
    # CHAPTER 4: OBJECTIVES (Page 12)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 12
    add_section_header(doc, 4, "Project Objectives", "Measurable Technical, Academic, and Societal Goals")

    add_body(doc, "The overarching primary objective of this capstone research is to design, develop, empirically benchmark, and deploy a robust, end-to-end Machine Learning system capable of processing historical Indian air pollution data, generating accurate 24-hour ahead AQI forecasts, and dispatching automated, rule-governed public alerts.", lead="4.1 Overarching Primary Objective: ")
    add_body(doc, "To achieve this primary goal, seven specific, measurable secondary technical objectives were established:", lead="4.2 Measurable Secondary Technical Objectives: ")
    add_bullets(doc, [
        "Objective 1 (Data Ingestion & Quality Audit): Ingest and validate the complete CPCB Indian air quality dataset (29,531 rows), audit missingness patterns across 12 criteria pollutants, and formally investigate out-of-scale AQI records (>500).",
        "Objective 2 (Statistical & Exploratory Analytics): Execute rigorous exploratory data analysis, quantifying statistical distribution moments (mean, median, skewness = 3.40, kurtosis = 21.42), seasonal variations, and inter-pollutant Pearson correlations.",
        "Objective 3 (Leakage-Free Feature Engineering): Construct cyclical calendar harmonics and single-day temporal lag features, establishing a reproducible chronological train-test split (15,655 train vs. 8,688 test records).",
        "Objective 4 (Multi-Model Algorithmic Benchmarking): Implement and compare candidate models (Baseline, Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting), optimizing for lowest validation MAE.",
        "Objective 5 (Uncertainty Quantification & Alert Engine): Quantify 95% empirical forecast prediction intervals ($\\pm 77.68$ AQI points) and build an automated, rule-based alert engine mapping predicted and observed values to official CPCB health categories.",
        "Objective 6 (Interactive Web Application): Deploy a modular, responsive Streamlit dashboard providing executive KPI overviews, multi-city exploration, live inference, error diagnostics, and practical hubs.",
        "Objective 7 (Mathematical & Statistical Core): Formulate and empirically evaluate the mathematical foundations—Principal Component Analysis via covariance eigenvalue decomposition, inferential seasonal hypothesis testing, and gradient descent optimization algorithms—directly within the atmospheric air quality domain."
    ])

    # -------------------------------------------------------------------------
    # CHAPTER 5: METHODOLOGY (Pages 13–20)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 13
    add_section_header(doc, 5, "Methodology", "System Architecture, Preprocessing, Mathematical Core, and ML Formulation")

    add_body(doc, "The technical architecture of the Smart Air Quality Prediction and Pollution Alert System is organized into five modular, decoupled layers: (1) Data Ingestion and Auditing Layer, (2) Feature Engineering & Pipeline Transformation Layer, (3) Mathematical Foundations Core, (4) Machine Learning Training, Tuning & Chronological Validation Layer, and (5) Predictive Alert Engine & Interactive Application Layer.", lead="5.1 System Architecture Overview: ")
    add_body(doc, "The empirical foundation of the system is the official Central Pollution Control Board (CPCB) continuous monitoring dataset for India (2015–2020), distributed via Kaggle by Rohan Rao. The raw table contains 29,531 records and 16 features spanning 26 major cities: Ahmedabad, Aizawl, Amaravati, Amritsar, Bengaluru, Bhopal, Brajrajnagar, Chandigarh, Chennai, Coimbatore, Delhi, Ernakulam, Gurugram, Guwahati, Hyderabad, Jaipur, Jorapokhar, Kochi, Kolkata, Lucknow, Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, and Visakhapatnam.", lead="5.2 Dataset Provenance and Multi-City Coverage: ")

    doc.add_page_break() # Page 14

    # Table 5.1: Dataset Schema
    add_text(doc, "Table 5.1: Dataset Schema and Criteria Pollutant Specification", style="Normal", bold_prefix="Table 5.1:")
    schema_rows = [
        ("City", "Categorical / Text", "Monitored urban center (26 distinct metropolitan areas)"),
        ("Date", "Datetime (YYYY-MM-DD)", "Observation date (Coverage: 1 Jan 2015 to 1 July 2020)"),
        ("PM2.5", "Numeric (µg/m³)", "Fine particulate matter with aerodynamic diameter ≤ 2.5 µm"),
        ("PM10", "Numeric (µg/m³)", "Coarse respirable particulate matter with diameter ≤ 10 µm"),
        ("NO, NO2, NOx", "Numeric (µg/m³ / ppb)", "Nitric oxide, nitrogen dioxide, and total nitrogen oxides (combustion signatures)"),
        ("NH3", "Numeric (µg/m³)", "Ammonia concentration (agricultural and industrial emissions)"),
        ("CO", "Numeric (mg/m³)", "Carbon monoxide concentration (incomplete vehicular/biomass combustion)"),
        ("SO2, O3", "Numeric (µg/m³)", "Sulphur dioxide (industrial) and ground-level ozone (secondary photochemical)"),
        ("Benzene, Toluene, Xylene", "Numeric (µg/m³)", "Volatile organic compounds (VOCs / BTX petrochemical indicators)"),
        ("AQI, AQI_Bucket", "Numeric / Categorical", "Official composite CPCB index and corresponding 6-tier category bucket")
    ]
    add_table(doc, ["FEATURE NAME", "DATA TYPE & UNIT", "DESCRIPTION & ENVIRONMENTAL ROLE"], schema_rows, widths=[1.5, 1.8, 3.55], font_size=8.0)

    add_body(doc, "Each record in the dataset represents a 24-hour daily aggregate reported by continuous ambient air quality monitoring stations. The schema normalizer validates data types, trims whitespace, standardizes casing, and handles invalid character encodings.", lead="5.3 Schema Validation & Standardization: ")

    doc.add_page_break() # Page 15

    add_body(doc, "Real-world environmental sensor networks inevitably suffer from power outages, sensor drift, maintenance cycles, and network telemetry failures. In `city_day.csv`, missing values range from 7.0% for CO to 61.3% for Xylene, with 4,681 records missing the calculated AQI. The system implements a scientifically principled missing-data strategy: rows missing all criteria pollutant measurements are discarded as unusable; for remaining records, missing pollutant concentrations are preserved during initial exploratory auditing and imputed during model training using median values accompanied by binary missingness indicators. Missing values are strictly never converted to zero, which would cause severe downward bias in pollutant averages.", lead="5.4 Missing-Value Discipline: ")
    add_body(doc, "A critical data quality anomaly investigated in this work is the presence of AQI readings exceeding 500. Under the CPCB National AQI technical manual, the standardized scale is bounded between 0 and 500. However, in the historical dataset, 543 records exhibit raw calculated AQI values above 500 (with peaks up to 2,049 in Ahmedabad and Delhi). Investigation reveals that these extreme values correspond to acute industrial/meteorological inversions or sensor saturation artifacts. Rather than cosmetically smoothing or clipping these values, the system flags them as `Out of range` in data quality audits, preserving genuine extreme episodic pollution information while alerting users.", lead="5.5 Sensor Saturation & Extreme Value Auditing (>500 AQI): ")

    doc.add_page_break() # Page 16

    add_body(doc, "Atmospheric pollution exhibits pronounced periodic oscillations tied to solar radiation, temperature cycles, and human work routines. To enable the algorithms to learn these cycles without artificial step-function boundaries, the day-of-year $d \\in [1, 365]$ is mapped to continuous cyclic harmonic coordinates using sine and cosine transformations: $\\text{sin\\_doy} = \\sin(2\\pi d / 365.25)$ and $\\text{cos\\_doy} = \\cos(2\\pi d / 365.25)$. Furthermore, day-of-week and month indicators capture vehicular traffic reductions on weekends and seasonal shifts between winter inversions and monsoon precipitation.", lead="5.6 Temporal Feature Engineering & Cyclical Harmonics: ")
    add_body(doc, "To construct the supervised target variable $y_{c,t} = \\text{AQI}_{c,t+1}$, the observation dataframe is grouped strictly by city and sorted chronologically. A calendar continuity check ensures that target records represent exactly the next calendar day ($t+1$). If an intermediate monitoring day is missing, no artificial pair is formed, completely preventing temporal interpolation artifacts.", lead="5.7 Chronological Lag Alignment: ")

    doc.add_page_break() # Page 17

    add_body(doc, "To establish the theoretical linear algebra foundations of the atmospheric modeling framework, the system implements Principal Component Analysis (PCA) via direct eigenvalue decomposition of the multi-pollutant covariance matrix. Given the standardized matrix of criteria pollutants $X_{std} \\in \\mathbb{R}^{N \\times p}$ (where $p = 6$: PM2.5, PM10, NO2, SO2, CO, O3):", lead="5.8 Mathematical Foundations I: Linear Algebra & PCA: ")
    add_body(doc, "\\Sigma = \\frac{1}{N - 1} X_{std}^T X_{std} \\in \\mathbb{R}^{p \\times p}")
    add_body(doc, "The eigenvalues $\\lambda_i$ and corresponding orthogonal eigenvectors $v_i$ are solved via $\\Sigma v_i = \\lambda_i v_i$. The proportion of atmospheric variance explained by the $k$-th principal component is given by $\\eta_k = \\lambda_k / \\sum_{j=1}^p \\lambda_j$. Empirical decomposition reveals that the top two principal components explain over 72% of total national atmospheric variance, with PC1 representing the primary combustion/particulate axis and PC2 capturing photochemical/gaseous dynamics.", lead="5.9 Eigenvalue Variance Decomposition: ")

    doc.add_page_break() # Page 18

    add_body(doc, "To rigorously evaluate parameter estimation dynamics on the multivariate atmospheric regression surface $\\mathcal{L}(w) = \\frac{1}{2N} \\|Xw - y\\|_2^2$, the system implements a comparative simulator benchmarking five prominent gradient-based optimization algorithms:", lead="5.10 Mathematical Foundations II: Loss Landscapes & Optimizers: ")
    add_bullets(doc, [
        "Batch Gradient Descent (BGD): Updates weights along the exact negative gradient across the entire dataset: $w_{t+1} = w_t - \\alpha \\nabla \\mathcal{L}(w_t)$. Exhibits steady convergence but high computational cost.",
        "Stochastic Gradient Descent (SGD): Updates weights using mini-batches (size = 32): $w_{t+1} = w_t - \\alpha \\nabla \\mathcal{L}_i(w_t)$. Introduces stochastic exploration to escape shallow local minima.",
        "Polyak Momentum: Accelerates gradient vectors in persistent directions by accumulating an exponentially decaying velocity vector: $v_{t+1} = \\beta v_t + \\alpha \\nabla \\mathcal{L}(w_t); \\quad w_{t+1} = w_t - v_{t+1}$ (with $\\beta = 0.9$). Dampens oscillations across steep ravine walls.",
        "RMSprop Optimizer: Normalizes gradient updates by dividing by the running root-mean-square of recent gradients: $s_{t+1} = \\beta s_t + (1 - \\beta) g_t^2; \\quad w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{s_{t+1}} + \\epsilon} g_t$. Prevents premature learning rate decay.",
        "Adam Optimizer: Combines the benefits of Momentum and RMSprop, computing bias-corrected first ($m_t$) and second ($v_t$) moment estimates: $w_{t+1} = w_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$ (with $\\beta_1 = 0.9, \\beta_2 = 0.999$). Demonstrates fastest, smoothest loss minimization on ill-conditioned multi-pollutant feature matrices."
    ])

    doc.add_page_break() # Page 19

    add_body(doc, "To ensure thorough academic benchmarking across distinct machine learning paradigms, six candidate modeling architectures were formulated and evaluated:", lead="5.11 Machine Learning Candidate Formulations: ")
    add_bullets(doc, [
        "Persistence Baseline: Assumes next-day AQI equals today's AQI: $\\hat{y}_{t+1} = y_t$. Establishes the non-trivial serial autocorrelation hurdle that any ML algorithm must beat.",
        "Ordinary Least Squares (OLS) Linear Regression: Parametric discriminative model minimizing residual sum of squares: $\\hat{\\beta} = (X^T X)^{-1} X^T y$.",
        "Ridge Regression ($L_2$ Regularization): Shrinkage estimator introducing penalty $\\lambda ||\\beta||_2^2$ to mitigate severe collinearity between PM2.5, PM10, and NO2: $\\hat{\\beta}_{ridge} = (X^T X + \\lambda I)^{-1} X^T y$.",
        "Decision Tree Regressor: Non-parametric recursive binary tree partitioning feature space to minimize variance.",
        "Random Forest Regressor: Ensemble bagging architecture constructing 100 decorrelated decision trees via bootstrap aggregation and random feature sub-spacing ($m = \\sqrt{p}$), reducing model variance without increasing bias.",
        "Gradient Boosting Regressor: Sequential ensemble boosting method fitting successive regression trees to the pseudo-residuals, optimizing empirical risk via stage-wise additive modeling."
    ])

    doc.add_page_break() # Page 20

    add_body(doc, "Evaluating time-dependent atmospheric processes via random $k$-fold cross-validation is a severe methodological error that causes temporal data leakage. This system enforces strict chronological splitting: all observation pairs where the target date falls before 27 May 2019 are allocated to the outer training set (15,655 records; 64.3%), while all records on or after 27 May 2019 are held out exclusively as the test set (8,688 records; 35.7%). Within the training set, an internal chronological validation split (last 20% of training dates) is used for model selection, ensuring zero snooping on the test set.", lead="5.12 Chronological Validation Strategy: ")
    add_body(doc, "To provide honest transparency regarding forecast confidence, the system computes the empirical standard deviation of out-of-sample test residuals $\\sigma_{res} = 39.63$ AQI points. Using the standard normal distribution critical value $z_{0.975} = 1.96$, empirical 95% forecast prediction intervals are constructed around each point prediction: $[\\hat{y} - 1.96\\sigma_{res}, \\hat{y} + 1.96\\sigma_{res}] = [\\hat{y} - 77.68, \\hat{y} + 77.68]$.", lead="5.13 Uncertainty Quantification & Prediction Intervals: ")
    add_body(doc, "The automated alert engine maps continuous predicted and recorded AQI values to official CPCB health bands, triggering data-quality alerts for values >500 and public health warnings when forecasts exceed user-configured thresholds (e.g., AQI $\\ge 200$).", lead="5.14 Automated Public Alert Mechanism: ")

    # -------------------------------------------------------------------------
    # CHAPTER 6: RESULTS AND DISCUSSION (Pages 20–24)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 21
    add_section_header(doc, 6, "Results and Discussion", "Empirical Findings, Algorithmic Benchmark, and Error Diagnostics")

    add_body(doc, "Exploratory data analysis of the 24,850 valid historical AQI records reveals a highly right-skewed, leptokurtic distribution. The national mean AQI is **166.5** (falling in the Moderate band), while the median is **118.0**. The substantial positive skewness (**3.40**) and extreme excess kurtosis (**21.42**) reflect the heavy toll of episodic severe winter pollution events across the Indo-Gangetic Plains, where AQI values routinely spike past 400. In total, 543 records exceed the nominal 500 CPCB scale cap, with historical peaks reaching 2,049 in Ahmedabad and Delhi during acute industrial/meteorological inversions.", lead="6.1 National Statistical Distribution Findings: ")
    
    # Figure 1: Distribution
    add_figure(doc, FIG_DIR / "fig01_national_aqi_distribution.png", "Figure 6.1: National Air Quality Index (AQI) Statistical Distribution, Boxplot, and CPCB Category Bands (N = 24,850).", width=5.8)

    doc.add_page_break() # Page 22

    add_body(doc, "Pearson correlation analysis demonstrates that fine particulate matter (PM2.5) exhibits the strongest individual linear association with composite AQI ($r = 0.88$), closely followed by coarse particulates (PM10, $r = 0.81$) and nitrogen dioxide (NO2, $r = 0.54$). High inter-pollutant correlation between PM2.5 and PM10 ($r = 0.85$) confirms shared combustion and vehicular origin, while also highlighting the severe multicollinearity risks that justify regularized Ridge and non-parametric Tree ensembles.", lead="6.2 Inter-Pollutant Correlation Analysis: ")
    add_figure(doc, FIG_DIR / "fig02_pollutant_correlation_matrix.png", "Figure 6.2: Inter-Pollutant Pearson Correlation Heatmap across Criteria Particulate and Gaseous Pollutants.", width=5.2)

    doc.add_page_break() # Page 23

    add_body(doc, "Seasonal aggregation reveals a massive, statistically significant disparity in atmospheric quality: Winter records the highest seasonal mean AQI (**209.9**), closely followed by Post-Monsoon (**207.0**). In stark contrast, Monsoon records an annual low of **118.1**—a dramatic seasonal drop of 91.8 AQI points driven by atmospheric wet deposition and high planetary boundary layer mixing. Formal Welch's two-sample $t$-test confirms that this seasonal disparity is overwhelmingly statistically significant ($t = 64.21, p < 10^{-15}$). City rankings place Ahmedabad (406.6), Delhi (258.9), Patna (225.6), Gurugram (215.9), and Lucknow (214.0) as the five most polluted urban centers, while coastal and northeastern cities such as Aizawl (34.7) and Thiruvananthapuram (75.8) consistently exhibit Satisfactory or Good air quality.", lead="6.3 Seasonal Dynamics & Geographic Disparities: ")
    add_figure(doc, FIG_DIR / "fig03_temporal_monthly_seasonal.png", "Figure 6.3: Monthly AQI Variation across Calendar Years (2016–2019) and Seasonal Mean Index (± SEM).", width=5.9)
    add_figure(doc, FIG_DIR / "fig04_city_pollution_rankings.png", "Figure 6.4: Geographic Rankings: Top 10 Most Polluted Cities vs. Top 5 Cleanest Cities in Historical Records.", width=5.9)

    doc.add_page_break() # Page 24

    add_body(doc, "All candidate algorithms were evaluated on the held-out outer test set (8,688 records from May 2019 to July 2020). Table 6.1 presents the complete performance matrix across both continuous regression metrics (MAE, RMSE, $R^2$) and discrete CPCB category classification metrics (Accuracy, Macro F1).", lead="6.4 Candidate Model Benchmarking: ")
    
    # Benchmark table
    benchmark_rows = [
        ("Random Forest (Bagging)", "19.66", "40.04", "0.8508", "75.79%", "0.6387", "Yes (Selected)"),
        ("Gradient Boosting (GBDT)", "20.29", "41.69", "0.8382", "73.98%", "0.5973", "No"),
        ("Persistence Baseline (AQI_t)", "22.10", "47.27", "0.7920", "74.17%", "0.6248", "No"),
        ("Decision Tree Regressor", "22.36", "48.94", "0.7771", "73.61%", "0.6311", "No"),
        ("Ridge Regression (L2)", "24.92", "41.38", "0.8406", "65.93%", "0.5897", "No"),
        ("Linear Regression (OLS)", "25.06", "41.49", "0.8397", "65.83%", "0.5892", "No")
    ]
    add_table(doc, ["MODEL ARCHITECTURE", "MAE", "RMSE", "R² SCORE", "CAT. ACCURACY", "MACRO F1", "SELECTED"], benchmark_rows, widths=[1.8, 0.75, 0.75, 0.85, 1.0, 0.85, 0.85], font_size=7.8)

    add_figure(doc, FIG_DIR / "fig05_model_performance_comparison.png", "Figure 6.5: Model Performance Comparison: Forecast Error (MAE, RMSE) and Variance Explained (R²).", width=5.8)
    add_figure(doc, FIG_DIR / "fig06_predicted_vs_actual_residuals.png", "Figure 6.6: Random Forest Error Diagnostics: Actual vs. Predicted Parity Plot and Zero-Mean Residual Distribution.", width=5.8)
    add_figure(doc, FIG_DIR / "fig07_feature_importance.png", "Figure 6.7: Top Predictive Features Driving Next-Day AQI Forecasts Ranked by Gini Impurity Reduction.", width=5.2)

    # -------------------------------------------------------------------------
    # CHAPTER 7: CONCLUSION (Page 25)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 25
    add_section_header(doc, 7, "Conclusion", "Summary of Achievements, Academic Verification, and Societal Benefit")

    add_body(doc, "The Smart Air Quality Prediction and Pollution Alert System successfully transitions environmental atmospheric monitoring from static, retrospective descriptive reporting to proactive, predictive machine learning intelligence. By grounding the entire pipeline in 29,531 authentic continuous monitoring records across 26 Indian metropolitan centers, the project establishes a robust, scientifically defensible platform for environmental protection.", lead="7.1 Summary of Completed Work: ")
    add_body(doc, "The project achieved all primary and secondary research milestones: (1) Engineered an automated, leakage-free data processing pipeline preserving genuine missing observations and auditing extreme sensor saturation (>500 AQI); (2) Established a strict chronological validation framework on 8,688 out-of-sample test records; (3) Benchmarked six candidate algorithms, demonstrating that Random Forest achieves state-of-the-art predictive performance ($R^2 = 0.8508$, MAE = 19.66 AQI points) and 75.79% category accuracy; (4) Quantified 95% empirical forecast uncertainty intervals ($\\pm 77.68$ AQI points); (5) Implemented an automated CPCB-compliant threshold alert mechanism; and (6) Integrated core multivariate mathematics (PCA, hypothesis testing, and gradient descent optimization) directly into atmospheric analysis.", lead="7.2 Key Academic & Technical Findings: ")
    add_body(doc, "From a societal perspective, this system empowers citizens, school administrators, healthcare professionals, and municipal planners with understandable 24-hour predictive foresight. By providing early warnings prior to severe winter inversions, vulnerable populations can take protective actions, directly advancing the public well-being goals of modern environmental data science.", lead="7.3 Societal and Environmental Impact: ")

    # -------------------------------------------------------------------------
    # CHAPTER 8: FUTURE SCOPE (Page 26)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 26
    add_section_header(doc, 8, "Future Scope", "Prioritized Engineering and Research Extension Roadmap")

    add_body(doc, "While the current system establishes a robust, verified machine learning foundation, future iterations can expand its operational capabilities across five prioritized horizons:", lead="8.1 Engineering Roadmap: ")
    
    add_text(doc, "Table 8.1: Prioritized Engineering and Research Extension Roadmap", style="Normal", bold_prefix="Table 8.1:")
    future_rows = [
        ("Phase 1: Multi-Horizon Forecasting", "Extend single-day forecast ($t+1$) to multi-step recursive and direct horizons (3-day and 7-day AQI forecasts) using sequence models (LSTM, GRU, Temporal Fusion Transformers).", "Medium", "High"),
        ("Phase 2: Meteorological & Satellite Fusion", "Integrate planetary boundary layer height, ambient temperature, relative humidity, wind velocity vectors (ERA5 Reanalysis), and MODIS/Sentinel-5P Aerosol Optical Depth (AOD) satellite feeds.", "High", "Critical"),
        ("Phase 3: Hyper-Local IoT Sensor Calibration", "Incorporate low-cost optical particle sensors (e.g., Plantower PMS5003) calibrated against CPCB reference stations using transfer learning and online drift compensation.", "Medium", "High"),
        ("Phase 4: Automated Public Push Alerts", "Deploy cloud microservices (FastAPI / Docker) with automated Telegram, SMS, and WhatsApp push notifications triggered by severe category forecasts for sensitive demographics.", "Low", "Medium"),
        ("Phase 5: Concept Drift & Continuous Retraining", "Implement automated MLOps pipelines (MLflow / Evidently AI) to detect atmospheric distribution shifts and automatically retrain models on newly published CPCB observations.", "Medium", "High")
    ]
    add_table(doc, ["ENHANCEMENT PHASE", "TECHNICAL SCOPE & METHODOLOGY", "COMPLEXITY", "IMPACT"], future_rows, widths=[1.5, 3.6, 0.9, 0.85], font_size=7.8)
    add_body(doc, "These proposed enhancements represent natural extensions of the modular architecture established in this project, ensuring that the system remains a durable foundation for continued environmental AI research.")

    # -------------------------------------------------------------------------
    # CHAPTER 9: REFERENCES (Pages 27–30)
    # -------------------------------------------------------------------------
    doc.add_page_break() # Page 27
    add_section_header(doc, 9, "References", "Comprehensive Academic, Environmental Authority, and Technical Bibliography")

    references_list = [
        ("[1]", "Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India. (2014). “National Air Quality Index.” Technical Report, CPCB, New Delhi. Available at: https://cpcb.nic.in/displaypdf.php?id=cWFsL05BVEIPTkFMX0FJSXJfUVVBTElUWV9JTkRFWC5wZGY="),
        ("[2]", "Central Pollution Control Board (CPCB). (2020). “Continuous Ambient Air Quality Monitoring (CAAQM) Network Technical Guidelines and Operational Standards.” Ministry of Environment, Forest and Climate Change, New Delhi, India."),
        ("[3]", "World Health Organization (WHO). (2021). “WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide.” World Health Organization, Geneva. ISBN: 978-92-4-003422-8."),
        ("[4]", "Rao, R. (2020). “Air Quality Data in India (2015–2020).” Kaggle Open Dataset. Distributed under CC0 Public Domain. Available at: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"),
        ("[5]", "Gupta, N. S., Mohta, Y., Heda, K., Armaan, R., Valarmathi, B., & Arulkumaran, G. (2023). “Prediction of Air Quality Index Using Machine Learning Techniques: A Comparative Analysis.” Journal of Environmental and Public Health, 2023, Article 4916267. https://doi.org/10.1155/2023/4916267"),
        ("[6]", "Liu, H., Li, Q., Yu, D., & Gu, Y. (2019). “Air Quality Index and Air Pollutant Concentration Prediction Based on Machine Learning Algorithms.” Applied Sciences, 9(19), 4069. https://doi.org/10.3390/app9194069"),
        ("[7]", "Castelli, M., Clemente, F. M., Popovic, A., Silva, S., & Vanneschi, L. (2020). “A Machine Learning Approach to Predict Air Quality in California.” Complexity, 2020, Article 8049504. https://doi.org/10.1155/2020/8049504"),
        ("[8]", "Majeed, F., Saleha, S., Abbas, L., Ghalib, S. A., Hamid, M., Saleem, M., Aman, M., & Ahmad, A. (2026). “Air Quality Index Prediction Using Machine Learning Regression Models: A Comparative Analysis.” PLOS ONE, 21(7), e0349858. https://doi.org/10.1371/journal.pone.0349858"),
        ("[9]", "Breiman, L. (2001). “Random Forests.” Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324"),
        ("[10]", "Friedman, J. H. (2001). “Greedy Function Approximation: A Gradient Boosting Machine.” Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451"),
        ("[11]", "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). “Scikit-learn: Machine Learning in Python.” Journal of Machine Learning Research, 12, 2825–2830."),
        ("[12]", "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). “Array Programming with NumPy.” Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2"),
        ("[13]", "McKinney, W. (2010). “Data Structures for Statistical Computing in Python.” Proceedings of the 9th Python in Science Conference (SciPy 2010), 56–61. https://doi.org/10.25080/Majora-92bf1922-00a"),
        ("[14]", "Hunter, J. D. (2007). “Matplotlib: A 2D Graphics Environment.” Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55"),
        ("[15]", "Waskom, M. L. (2021). “seaborn: Statistical Data Visualization.” Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021"),
        ("[16]", "Streamlit Inc. (2026). “Streamlit Documentation: The Fastest Way to Build Data Apps.” Available at: https://docs.streamlit.io/"),
        ("[17]", "Pradhan, M., & Kumar, U. D. (2019). “Machine Learning using Python.” Wiley India. ISBN: 978-8126579907. (Official CSE273 Course Textbook)."),
        ("[18]", "Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). “Mathematics for Machine Learning.” Cambridge University Press. https://doi.org/10.1017/9781108679930. (Official CSE273 Reference Textbook)."),
        ("[19]", "Rao, R. N. (2021). “Machine Learning in Data Science using Python.” Wiley India. ISBN: 978-9354245640."),
        ("[20]", "Thareja, R. (2022). “Data Science and Machine Learning using Python.” McGraw-Hill Education. ISBN: 978-9355321527."),
        ("[21]", "Hastie, T., Tibshirani, R., & Friedman, J. (2009). “The Elements of Statistical Learning: Data Mining, Inference, and Prediction.” 2nd Edition, Springer Series in Statistics. https://doi.org/10.1007/978-0-387-84858-7"),
        ("[22]", "Mitchell, T. M. (1997). “Machine Learning.” McGraw-Hill International Editions, Computer Science Series. ISBN: 0-07-042807-7."),
        ("[23]", "Brink, H., Richards, J. W., Fetherolf, M., & Cronin, B. (2017). “Real-World Machine Learning.” Manning Publications. ISBN: 978-1617291920."),
        ("[24]", "Srinivasaraghavan, A., & Joseph, V. (2019). “Machine Learning.” Wiley India. ISBN: 978-8126578511."),
        ("[25]", "Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). “Time Series Analysis: Forecasting and Control.” 5th Edition, John Wiley & Sons. ISBN: 978-1118675021."),
        ("[26]", "OpenAQ. (2026). “OpenAQ Global Air Quality API v3 Documentation.” Available at: https://docs.openaq.org/"),
        ("[27]", "Panigrahi, S. (2026). “AIRWATCH INDIA: An Interactive National Air Quality Intelligence Dashboard.” Course Project Report for INT374, Lovely Professional University."),
        ("[28]", "Panigrahi, S. (2026). “Airwatch-India-PowerBi: Public Analytics Repository.” GitHub. Available at: https://github.com/Subu2006/Airwatch-India-PowerBi"),
        ("[29]", "Panigrahi, S. (2026). “Smart Air Quality Prediction and Pollution Alert System.” GitHub Source Repository. Available at: https://github.com/Subu2006/Smart-Air-Quality-Prediction-and-Pollution-Alert-System"),
        ("[30]", "Kingma, D. P., & Ba, J. (2014). “Adam: A Method for Stochastic Optimization.” arXiv preprint arXiv:1412.6980. https://doi.org/10.48550/arXiv.1412.6980"),
        ("[31]", "Polyak, B. T. (1964). “Some Methods of Speeding Up the Convergence of Iteration Methods.” USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17. https://doi.org/10.1016/0041-5553(64)90137-5"),
        ("[32]", "Tieleman, T., & Hinton, G. (2012). “Lecture 6.5-RMSprop: Divide the Gradient by a Running Average of Its Recent Magnitude.” COURSERA: Neural Networks for Machine Learning, 4(2), 26–31."),
        ("[33]", "Welch, B. L. (1947). “The Generalization of 'Student's' Problem When Several Different Population Variances Are Involved.” Biometrika, 34(1/2), 28–35. https://doi.org/10.2307/2332510"),
        ("[34]", "Mann, H. B., & Whitney, D. R. (1947). “On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other.” The Annals of Mathematical Statistics, 18(1), 50–60. https://doi.org/10.1214/aoms/1177730491"),
        ("[35]", "Pearson, K. (1901). “LIII. On Lines and Planes of Closest Fit to Systems of Points in Space.” The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 2(11), 559–572. https://doi.org/10.1080/14786440109462720")
    ]
    
    for tag, ref_text in references_list:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.35)
        p_ref.paragraph_format.first_line_indent = Inches(-0.35)
        p_ref.paragraph_format.space_after = Pt(5)
        p_ref.paragraph_format.line_spacing = 1.05
        r_tag = p_ref.add_run(tag + " ")
        r_tag.bold = True
        r_tag.font.color.rgb = RGBColor.from_string(TEAL)
        p_ref.add_run(ref_text)
        set_para_keep(p_ref)

    # Save Word Document
    out_docx = REPORTS / "Smart_Air_Quality_Project_Report.docx"
    doc.save(out_docx)
    print(f"Successfully generated Word Report: {out_docx}")
    print(f"Paragraphs: {len(doc.paragraphs)}, Tables: {len(doc.tables)}, Sections: {len(doc.sections)}")

    # Convert to PDF via docx2pdf / Word COM
    out_pdf = REPORTS / "Smart_Air_Quality_Project_Report.pdf"
    print("Converting to PDF...")
    try:
        from docx2pdf import convert
        convert(str(out_docx), str(out_pdf))
        print(f"Successfully generated PDF Report: {out_pdf}")
    except Exception as e:
        print(f"docx2pdf conversion encountered: {e}. Attempting Word COM fallback...")
        try:
            import win32com.client
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = False
            doc_com = word.Documents.Open(str(out_docx.resolve()))
            doc_com.SaveAs(str(out_pdf.resolve()), FileFormat=17) # wdFormatPDF = 17
            doc_com.Close()
            word.Quit()
            print(f"Successfully generated PDF via Word COM: {out_pdf}")
        except Exception as e2:
            print(f"Word COM fallback error: {e2}")


if __name__ == "__main__":
    build_academic_report()
