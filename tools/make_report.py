import json
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

LAYOUT = 'report_assets/doc_layout.json'
TEMPLATE = 'report_assets/Никончук_11b_11var_backup.docx'
OUTPUT = 'Шульдов_10a_4var.docx'

layout = json.load(open(LAYOUT, encoding='utf-8'))
doc = docx.Document(TEMPLATE)


def clear_hf(hf):
    root = hf._element
    for child in list(root):
        if child.tag in (qn('w:p'), qn('w:tbl')):
            root.remove(child)
    root.append(OxmlElement('w:p'))


def add_page_number(footer):
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement('w:fldChar')
    fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = 'PAGE'
    fld2 = OxmlElement('w:fldChar')
    fld2.set(qn('w:fldCharType'), 'end')
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)


for sec in doc.sections:
    clear_hf(sec.header)
    clear_hf(sec.first_page_header)
    clear_hf(sec.even_page_header)
    clear_hf(sec.footer)
    clear_hf(sec.first_page_footer)
    clear_hf(sec.even_page_footer)
    add_page_number(sec.footer)
    add_page_number(sec.first_page_footer)

body = doc.element.body
for child in list(body):
    if child.tag != qn('w:sectPr'):
        body.remove(child)


def set_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '4')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    tblPr.append(borders)


def set_cell_text(cell, text, bold=False, size=None):
    cell.text = ''
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    if size:
        run.font.size = Pt(size)


def add_grid_table(rows, widths=None, header=True, font_size=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    set_borders(t)
    t.autofit = False
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell_text(t.cell(i, j), str(val), bold=header and i == 0, size=font_size)
            if widths:
                t.cell(i, j).width = Inches(widths[j])
    return t


def add_p(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, size=None, italic=False):
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_picture(fname, width_in, caption):
    doc.add_picture(f'report_assets/{fname}', width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_p(caption, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9)


def add_calc_box(text):
    t = doc.add_table(rows=1, cols=1)
    set_borders(t)
    set_cell_text(t.cell(0, 0), text)
    t.cell(0, 0).width = Inches(7.0)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)


for e in layout['elements']:
    t = e['t']
    if t == 'title':
        for i, line in enumerate(e['lines']):
            add_p(line, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16 if i == 0 else 14)
        add_p('', align=WD_ALIGN_PARAGRAPH.CENTER)
    elif t == 'table':
        add_grid_table(e['rows'], widths=[2.3, 2.0, 1.4])
        add_p('', align=WD_ALIGN_PARAGRAPH.CENTER)
    elif t == 'summary':
        add_p('Сводная таблица испытаний', bold=True)
        add_grid_table(e['rows'], widths=[0.45, 1.05, 1.25, 1.95, 0.7, 1.2], font_size=8)
    elif t == 'auto':
        add_grid_table(e['rows'], widths=[1.5, 4.6, 0.8], font_size=8)
    elif t == 'h1':
        doc.add_page_break() if e['text'][0].isdigit() else None
        doc.add_heading(e['text'], level=1)
    elif t == 'h2':
        doc.add_heading(e['text'], level=2)
    elif t == 'p':
        add_p(e['text'])
    elif t == 'calc':
        add_calc_box(e['text'])
    elif t == 'img':
        add_picture(e['file'], e.get('w', 6.4), e.get('cap', ''))

doc.save(OUTPUT)
print(f'saved: {OUTPUT}')
