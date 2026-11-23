# -*- coding: utf-8 -*-
"""Builds an editable PowerPoint slide (native chart, not an image) for the
model-comparison bar chart (Fig. 2 in the paper), so it can be tweaked directly
in PowerPoint: colors, labels, title, data values, chart type, etc."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import numpy as np

models = [
    "GCN (baseline)",
    "GATv2 (PAE-w., tuned)",
    "GATv2 (unweighted, tuned)",
    "XGBoost (no structure)",
    "GATv2 (PAE-w., original)",
    "GATv2 (unweighted, original)",
    "GraphSAGE (baseline)",
    "AlphaMissense",
]
auprc = [0.8938, 0.9069, 0.9097, 0.9125, 0.9112, 0.9112, 0.9148, 0.9286]

order = np.argsort(auprc)
models = [models[i] for i in order]
auprc = [auprc[i] for i in order]

# Highlight colors: proposed model (PAE-weighted, tuned) and the AlphaMissense
# benchmark stand out; everything else stays a neutral grey.
COL_DEFAULT = RGBColor(0x8B, 0x97, 0x9E)
COL_PROPOSED = RGBColor(0xB5, 0x49, 0x3D)
COL_ALPHAMISSENSE = RGBColor(0x2E, 0x6F, 0x8E)

point_colors = []
for m in models:
    if "PAE-w., tuned" in m:
        point_colors.append(COL_PROPOSED)
    elif m == "AlphaMissense":
        point_colors.append(COL_ALPHAMISSENSE)
    else:
        point_colors.append(COL_DEFAULT)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

# Title
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.9))
tf = title_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Model comparison on the ClinVar missense test set (n = 6,771)"
p.font.size = Pt(28)
p.font.bold = True
p.font.name = "Calibri"
p.font.color.rgb = RGBColor(0x1B, 0x22, 0x2B)

# Subtitle
sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.05), Inches(12.333), Inches(0.5))
tf2 = sub_box.text_frame
p2 = tf2.paragraphs[0]
p2.text = "Test-set AUPRC, all seven trained models plus AlphaMissense"
p2.font.size = Pt(16)
p2.font.italic = True
p2.font.color.rgb = RGBColor(0x57, 0x64, 0x6D)

chart_data = CategoryChartData()
chart_data.categories = models
chart_data.add_series("Test-set AUPRC", auprc)

x, y, cx, cy = Inches(0.6), Inches(1.7), Inches(12.1), Inches(5.4)
graphic_frame = slide.shapes.add_chart(
    XL_CHART_TYPE.BAR_CLUSTERED, x, y, cx, cy, chart_data
)
chart = graphic_frame.chart
chart.has_legend = False
chart.has_title = False

plot = chart.plots[0]
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = "0.0000"
dl.number_format_is_linked = False
dl.font.size = Pt(13)
dl.font.color.rgb = RGBColor(0x1B, 0x22, 0x2B)
dl.position = 2  # xlLabelPositionOutsideEnd (BAR chart)

series = plot.series[0]
for i, pt in enumerate(series.points):
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = point_colors[i]
    pt.format.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    pt.format.line.width = Pt(0.75)

value_axis = chart.value_axis
value_axis.minimum_scale = 0.88
value_axis.maximum_scale = 0.935
value_axis.major_unit = 0.01
value_axis.tick_labels.font.size = Pt(12)
value_axis.has_major_gridlines = True
value_axis.format.line.color.rgb = RGBColor(0xB0, 0xB0, 0xB0)

category_axis = chart.category_axis
category_axis.tick_labels.font.size = Pt(13)
category_axis.tick_labels.font.bold = False

prs.save("model_comparison_slide.pptx")
print("saved model_comparison_slide.pptx")
