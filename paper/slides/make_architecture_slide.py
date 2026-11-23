# -*- coding: utf-8 -*-
"""Recreates the 5-panel architecture-comparison figure (Fig. 1 in the paper,
"How each model turns a protein into a pathogenicity prediction") as native,
individually-selectable PowerPoint shapes, so it can be edited directly in
PowerPoint: move/recolor/resize any node, edge, box, or text independently."""

import math
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn

# ---------- design tokens (matched to figures/architecture_comparison.png) ----------
COL_NODE = RGBColor(0x93, 0xA8, 0xB3)
COL_NODE_EDGE = RGBColor(0x5B, 0x7A, 0x88)
COL_MUT = RGBColor(0xE0, 0x70, 0x3C)
COL_MUT_EDGE = RGBColor(0xB4, 0x4F, 0x22)
COL_EDGE_FLAT = RGBColor(0xB9, 0xC4, 0xC8)
COL_HERO_BORDER = RGBColor(0x2E, 0x6F, 0x8E)
COL_ABLATION_BORDER = RGBColor(0xB0, 0x7A, 0x24)
COL_PANEL_BG_HERO = RGBColor(0xF4, 0xF9, 0xFB)
COL_PANEL_BG_ABL = RGBColor(0xFB, 0xF7, 0xEF)
COL_BOX_BG = RGBColor(0xEF, 0xF3, 0xF4)
COL_BOX_EDGE = RGBColor(0x8F, 0xA6, 0xB2)
COL_TEXT = RGBColor(0x1B, 0x22, 0x2B)
COL_TEXT_MUTED = RGBColor(0x57, 0x64, 0x6D)
COL_ARROW = RGBColor(0x57, 0x64, 0x6D)
COL_WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank


def set_no_line(shape):
    shape.line.fill.background()


def add_textbox(x, y, w, h, text, size, color=COL_TEXT, bold=False, italic=False,
                 align=PP_ALIGN.CENTER, font=FONT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.italic = italic
        p.font.name = font
        p.font.color.rgb = color
    return tb


def add_rounded_box(x, y, w, h, text, fill_color, border_color, size=10, textcolor=COL_TEXT):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.adjustments[0] = 0.12
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.color.rgb = border_color
    box.line.width = Pt(1.25)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(size)
        p.font.name = FONT
        p.font.color.rgb = textcolor
    return box


def add_down_arrow(cx, y_top, h, color=COL_ARROW, w=Inches(0.16)):
    shp = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, cx - w / 2, y_top, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    set_no_line(shp)
    return shp


def add_line(x1, y1, x2, y2, color, width_pt):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = color
    conn.line.width = Pt(width_pt)
    return conn


def add_oval(cx, cy, d, fill_color, edge_color, edge_w=1.0):
    o = slide.shapes.add_shape(MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d)
    o.fill.solid()
    o.fill.fore_color.rgb = fill_color
    o.line.color.rgb = edge_color
    o.line.width = Pt(edge_w)
    return o


# ---------- ring-graph geometry (8 residues, shared by all graph-based panels) ----------
N_RING = 8
MUT_IDX = 0
ring_edges = [(i, (i + 1) % N_RING) for i in range(N_RING)]
chord_edges = [(1, 4), (2, 6), (5, 7)]
all_edges = ring_edges + chord_edges
# synthetic confidence weights for the PAE-weighted panel only (low = long-range, high = local)
edge_weight = {}
_local_w = [0.62, 0.71, 0.58, 0.93, 0.66, 0.81, 0.60, 0.88]
_chord_w = [0.22, 0.33, 0.18]
for e, w in zip(ring_edges, _local_w):
    edge_weight[e] = w
for e, w in zip(chord_edges, _chord_w):
    edge_weight[e] = w


def ring_positions(cx, cy, r):
    pos = {}
    for i in range(N_RING):
        angle = math.pi / 2 - i * (2 * math.pi / N_RING)  # node 0 at top, clockwise
        pos[i] = (cx + r * math.cos(angle), cy - r * math.sin(angle))
    return pos


def draw_ring_graph(cx, cy, r, weighted):
    pos = ring_positions(cx, cy, r)
    for e in all_edges:
        (x1, y1), (x2, y2) = pos[e[0]], pos[e[1]]
        if weighted:
            w = edge_weight[e]
            width_pt = 0.6 + 3.0 * w
            # blend flat-grey -> hero-blue as weight increases
            t = w
            r_c = int(COL_EDGE_FLAT[0] + t * (COL_HERO_BORDER[0] - COL_EDGE_FLAT[0]))
            g_c = int(COL_EDGE_FLAT[1] + t * (COL_HERO_BORDER[1] - COL_EDGE_FLAT[1]))
            b_c = int(COL_EDGE_FLAT[2] + t * (COL_HERO_BORDER[2] - COL_EDGE_FLAT[2]))
            add_line(Inches(x1), Inches(y1), Inches(x2), Inches(y2),
                     RGBColor(r_c, g_c, b_c), width_pt)
        else:
            add_line(Inches(x1), Inches(y1), Inches(x2), Inches(y2), COL_EDGE_FLAT, 1.5)
    for i, (x, y) in pos.items():
        if i == MUT_IDX:
            add_oval(Inches(x), Inches(y), Inches(0.22), COL_MUT, COL_MUT_EDGE, 1.25)
        else:
            add_oval(Inches(x), Inches(y), Inches(0.16), COL_NODE, COL_NODE_EDGE, 1.0)


def draw_vector_icon(cx, cy):
    halo = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.45), Inches(cy - 0.45),
                                   Inches(0.9), Inches(0.9))
    halo.fill.background()
    halo.line.color.rgb = COL_EDGE_FLAT
    halo.line.width = Pt(1.25)
    halo.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    add_oval(Inches(cx), Inches(cy), Inches(0.22), COL_MUT, COL_MUT_EDGE, 1.25)


# ---------- panel layout ----------
N_PANELS = 5
LEFT_MARGIN = Inches(0.3)
GAP = Inches(0.16)
PANEL_W = Inches((13.333 - 0.6 - 4 * 0.16) / 5)

Y_TAG = Inches(1.00)
Y_TITLE = Inches(1.27)
Y_SUBTITLE = Inches(1.62)
GRAPH_CY_IN = 2.85          # inches, absolute y of ring center
GRAPH_R_IN = 0.55           # inches, ring radius
Y_ARROW1 = Inches(3.58)
Y_LAYERBOX = Inches(3.84)
H_LAYERBOX = Inches(0.55)
Y_ARROW2 = Inches(4.45)
Y_READOUT = Inches(4.66)
Y_ARROW3 = Inches(5.06)
Y_CLASSIFIER = Inches(5.28)
H_CLASSIFIER = Inches(0.38)
Y_ARROW4 = Inches(5.72)
Y_PCIRCLE = Inches(5.95)
D_PCIRCLE = Inches(0.46)
Y_LABEL = Inches(6.48)

panels = [
    dict(title="XGBoost", subtitle="structure-free baseline", kind="vector",
         layer_text="XGBoost\n(gradient-boosted trees)", tag=None),
    dict(title="GCN", subtitle="graph baseline, no weighting", kind="flat",
         layer_text="graph layers x2\n(no attention)", tag=None),
    dict(title="GraphSAGE", subtitle="graph baseline, no weighting", kind="flat",
         layer_text="graph layers x2\n(neighbour averaging)", tag=None),
    dict(title="GATv2 (unweighted)", subtitle="attention used, PAE ignored", kind="flat_gat",
         layer_text="GATv2 layers x2\nedge_dim = None", tag="ABLATION"),
    dict(title="GATv2 (PAE-weighted)", subtitle="attention guided by AlphaFold confidence",
         kind="weighted", layer_text="GATv2 layers x2\nedge_dim = 2", tag="PROPOSED MODEL"),
]

# Title
add_textbox(Inches(0.3), Inches(0.18), Inches(12.7), Inches(0.6),
            "How each model turns a protein into a pathogenicity prediction",
            22, color=COL_TEXT, bold=True)

for i, p in enumerate(panels):
    x0 = LEFT_MARGIN + i * (PANEL_W + GAP)
    cx_in = (x0 + PANEL_W / 2) / Emu(1) * 1  # not used; compute cx in inches directly below
    cx_in = (x0.inches if hasattr(x0, "inches") else x0 / 914400) + PANEL_W.inches / 2

    hero = p["tag"] == "PROPOSED MODEL"
    ablation = p["tag"] == "ABLATION"

    # outer highlight panel background + border for the two central-comparison models
    if hero or ablation:
        bg = COL_PANEL_BG_HERO if hero else COL_PANEL_BG_ABL
        border = COL_HERO_BORDER if hero else COL_ABLATION_BORDER
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x0 - Inches(0.05),
                                       Inches(0.92), PANEL_W + Inches(0.1), Inches(5.9))
        rect.adjustments[0] = 0.03
        rect.fill.solid()
        rect.fill.fore_color.rgb = bg
        rect.line.color.rgb = border
        rect.line.width = Pt(1.75)
        add_textbox(x0, Y_TAG, PANEL_W, Inches(0.2), p["tag"], 10, color=border, bold=True)

    add_textbox(x0, Y_TITLE, PANEL_W, Inches(0.3), p["title"], 13, bold=True)
    add_textbox(x0, Y_SUBTITLE, PANEL_W, Inches(0.45), p["subtitle"], 9,
                color=COL_TEXT_MUTED, italic=True)

    if p["kind"] == "vector":
        draw_vector_icon(cx_in, GRAPH_CY_IN)
        add_textbox(x0, Inches(GRAPH_CY_IN - 0.15), PANEL_W, Inches(0.6),
                    "no 3D structure used", 8.5, color=COL_TEXT_MUTED, italic=True)
    else:
        draw_ring_graph(cx_in, GRAPH_CY_IN, GRAPH_R_IN, weighted=(p["kind"] == "weighted"))

    add_down_arrow(x0 + PANEL_W / 2, Y_ARROW1, Inches(0.22))

    box_fill = COL_PANEL_BG_HERO if hero else (COL_PANEL_BG_ABL if ablation else COL_BOX_BG)
    box_border = COL_HERO_BORDER if hero else (COL_ABLATION_BORDER if ablation else COL_BOX_EDGE)
    add_rounded_box(x0 + Inches(0.06), Y_LAYERBOX, PANEL_W - Inches(0.12), H_LAYERBOX,
                    p["layer_text"], box_fill, box_border, size=9.5)

    add_down_arrow(x0 + PANEL_W / 2, Y_ARROW2, Inches(0.18))

    if p["kind"] != "vector":
        add_textbox(x0, Y_READOUT, PANEL_W, Inches(0.35),
                    "read out mutated residue's node only", 8, color=COL_TEXT_MUTED)

    add_down_arrow(x0 + PANEL_W / 2, Y_ARROW3, Inches(0.18))

    add_rounded_box(x0 + Inches(0.55), Y_CLASSIFIER, PANEL_W - Inches(1.1), H_CLASSIFIER,
                    "classifier", COL_BOX_BG, COL_BOX_EDGE, size=9.5)

    add_down_arrow(x0 + PANEL_W / 2, Y_ARROW4, Inches(0.18))

    pcx = x0 + PANEL_W / 2
    circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, pcx - D_PCIRCLE / 2, Y_PCIRCLE,
                                   D_PCIRCLE, D_PCIRCLE)
    circ.fill.solid()
    circ.fill.fore_color.rgb = COL_WHITE
    circ.line.color.rgb = COL_TEXT
    circ.line.width = Pt(1.25)
    tf = circ.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    p_ = tf.paragraphs[0]
    p_.text = "P"
    p_.alignment = PP_ALIGN.CENTER
    p_.font.size = Pt(11)
    p_.font.italic = True
    p_.font.color.rgb = COL_TEXT

    add_textbox(x0, Y_LABEL, PANEL_W, Inches(0.35), "pathogenic or benign", 8.5,
                color=COL_TEXT_MUTED)

# ---------- legend ----------
leg_y = Inches(7.05)
leg_items_x = [0.8, 3.7, 6.6, 9.6]
add_oval(Inches(leg_items_x[0]), leg_y + Inches(0.09), Inches(0.16), COL_NODE, COL_NODE_EDGE)
add_textbox(Inches(leg_items_x[0] + 0.18), leg_y, Inches(2.4), Inches(0.3),
            "ordinary residue", 9.5, align=PP_ALIGN.LEFT)

add_oval(Inches(leg_items_x[1]), leg_y + Inches(0.09), Inches(0.16), COL_MUT, COL_MUT_EDGE)
add_textbox(Inches(leg_items_x[1] + 0.18), leg_y, Inches(2.7), Inches(0.3),
            "mutated residue (evaluated)", 9.5, align=PP_ALIGN.LEFT)

add_line(Inches(leg_items_x[2]), leg_y + Inches(0.09), Inches(leg_items_x[2] + 0.35),
         leg_y + Inches(0.09), COL_EDGE_FLAT, 2.2)
add_textbox(Inches(leg_items_x[2] + 0.45), leg_y, Inches(2.2), Inches(0.3),
            "contact, equal weight", 9.5, align=PP_ALIGN.LEFT)

add_line(Inches(leg_items_x[3]), leg_y + Inches(0.09), Inches(leg_items_x[3] + 0.35),
         leg_y + Inches(0.09), COL_HERO_BORDER, 3.2)
add_textbox(Inches(leg_items_x[3] + 0.45), leg_y, Inches(3.2), Inches(0.4),
            "contact, weight ~ AlphaFold confidence\n(thicker/darker = more confident)", 9,
            align=PP_ALIGN.LEFT)

prs.save("architecture_comparison_slide.pptx")
print("saved architecture_comparison_slide.pptx")
