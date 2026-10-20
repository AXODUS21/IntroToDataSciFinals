"""
Converts FINAL_PROJECT_DOCUMENTATION.md into a beautifully formatted Microsoft Word (.docx) document
with embedded high-resolution figures, styled tables, headings, and formatting.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, hex_color):
    """Sets background shading of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def create_word_report():
    doc = Document()

    # Set margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Document Title Block
    title = doc.add_paragraph()
    title_run = title.add_run("DE LA SALLE LIPA\n")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(14)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(16, 124, 65) # Lasallian Green

    sub = title.add_run("College of Information Technology and Engineering\nComputer Science Department\n")
    sub.font.name = "Arial"
    sub.font.size = Pt(11)
    sub.font.bold = True
    sub.font.color.rgb = RGBColor(80, 80, 80)

    p_proj = doc.add_paragraph()
    p_proj_run = p_proj.add_run("FINAL PROJECT REPORT: INTRODUCTION TO DATA SCIENCE (INTDSCI)\nSupervised Binary Classification of Hourly Bike Rental Demand in Seoul")
    p_proj_run.font.name = "Arial"
    p_proj_run.font.size = Pt(16)
    p_proj_run.font.bold = True
    p_proj_run.font.color.rgb = RGBColor(30, 30, 30)

    doc.add_paragraph("Dataset: Seoul Bike Sharing Demand Dataset (UCI ML Repository)\nDeliverables: Python Code Pipeline, Reproducible Jupyter Notebook, & Project Report")
    doc.add_paragraph("-" * 80)

    # Reading markdown source
    md_path = os.path.join("reports", "FINAL_PROJECT_DOCUMENTATION.md")
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    table_rows = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        
        num_cols = len(table_rows[0])
        t = doc.add_table(rows=len(table_rows), cols=num_cols)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.style = 'Table Grid'
        
        for r_idx, row in enumerate(table_rows):
            for c_idx, cell_value in enumerate(row):
                cell = t.cell(r_idx, c_idx)
                cell.text = cell_value.strip().replace("**", "").replace("*", "")
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                if r_idx == 0:
                    set_cell_background(cell, "2E7D32") # Dark green header
                    for run in p.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(9.5)
                else:
                    if r_idx % 2 == 1:
                        set_cell_background(cell, "F5F5F5")
                    for run in p.runs:
                        run.font.size = Pt(9)
        doc.add_paragraph()
        table_rows = []
        in_table = False

    for line in lines:
        stripped = line.strip()

        # Handle Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            parts = [p.strip() for p in stripped[1:-1].split("|")]
            # Skip separator line like | :--- | :--- |
            if all(set(p).issubset({'-', ':', ' '}) for p in parts if p):
                continue
            table_rows.append(parts)
            in_table = True
            continue
        elif in_table:
            flush_table()

        if not stripped:
            continue

        # Skip main markdown title since we custom formatted title block
        if stripped.startswith("# ") and "Comprehensive Project Documentation" in stripped:
            continue
        if stripped.startswith("## ") and "Introduction to Data Science" in stripped:
            continue

        # Handle Headings
        if stripped.startswith("# "):
            h = doc.add_heading(stripped[2:], level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)
            for r in h.runs:
                r.font.color.rgb = RGBColor(16, 124, 65)
        elif stripped.startswith("## "):
            h = doc.add_heading(stripped[3:], level=2)
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
            for r in h.runs:
                r.font.color.rgb = RGBColor(33, 33, 33)
        elif stripped.startswith("### "):
            h = doc.add_heading(stripped[4:], level=3)
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(3)
        elif stripped.startswith("#### "):
            h = doc.add_heading(stripped[5:], level=4)
            h.paragraph_format.space_before = Pt(6)
            h.paragraph_format.space_after = Pt(2)
        elif stripped.startswith("!["):
            # Image embed format: ![Caption](file:///path/to/fig.png)
            try:
                caption = stripped[2:stripped.index("]")]
                img_url = stripped[stripped.index("(") + 1 : stripped.index(")")]
                img_path = img_url.replace("file://", "")
                if os.path.exists(img_path):
                    doc.add_paragraph()
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    doc.add_picture(img_path, width=Inches(5.5))
                    p_cap = doc.add_paragraph(f"Caption: {caption}")
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap.runs[0].font.italic = True
                    p_cap.runs[0].font.size = Pt(9)
            except Exception as e:
                print(f"Image embed error: {e}")
        elif stripped.startswith("```"):
            # Skip code fence markers
            continue
        elif stripped.startswith("* ") or stripped.startswith("- "):
            p = doc.add_paragraph(style='List Bullet')
            text = stripped[2:].replace("**", "")
            p.add_run(text)
            p.paragraph_format.space_after = Pt(2)
        elif stripped[0].isdigit() and stripped[1:3] in [". ", ") "]:
            p = doc.add_paragraph(style='List Number')
            text = stripped[3:].replace("**", "")
            p.add_run(text)
            p.paragraph_format.space_after = Pt(2)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            p.add_run(stripped.replace("**", ""))

    if in_table:
        flush_table()

    output_docx = os.path.join("reports", "FINAL_PROJECT_DOCUMENTATION.docx")
    doc.save(output_docx)
    print(f"Successfully generated Word document at {output_docx}!")

if __name__ == "__main__":
    create_word_report()
