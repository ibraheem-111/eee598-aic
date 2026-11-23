#!/usr/bin/env python3
"""Build the combined EEE 598 deck in the ASU 2026 style from the official template (assets only).

Sources: the AIC progress deck (Ibraheem), Chinmay's slides 9-16 (PDF), the control slides (pptx).
Two frames from the template: cover/closing (network art, layout of template slide 8) and content (white,
gold-highlight title, layout of template slide 12). Everything else is drawn per slide with native shapes,
tables and charts. Run once; afterwards the .pptx is edited in place, never rebuilt over.
"""
import copy
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "asu/template.pptx"
ICONS = HERE / "icons"
SRC = HERE / "src"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out.pptx"

MAROON, GOLD, INK, GRAY, PANEL, WHITE = (RGBColor(0x8C, 0x1D, 0x40), RGBColor(0xFF, 0xC6, 0x27),
                                          RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x74, 0x74, 0x74),
                                          RGBColor(0xF2, 0xF2, 0xF2), RGBColor(0xFF, 0xFF, 0xFF))
LIGHTGOLD = RGBColor(0xFF, 0xF1, 0xC9)
MIDGRAY = RGBColor(0xBF, 0xBF, 0xBF)
FONT = "Arial"
L, R, TOP, BOT = 0.37, 9.63, 1.0, 4.95  # content area (inches)

prs = Presentation(TEMPLATE)
ORIG = list(prs.slides._sldIdLst)
COVER = next(l for l in prs.slide_layouts if l.name == "CUSTOM_23_1_1_1_1_1_1_2_1_2")
CONTENT = next(l for l in prs.slide_layouts if l.name == "CUSTOM_40")


# ------------------------------------------------------------------ helpers
def nostyle(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)
    return shape


def _run(p, text, size, bold, color, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color
    return r


def text(slide, x, y, w, h, content, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=4, bullets=False):
    """content: str, or list of paragraphs; a paragraph is str or list of (text, bold[, color]) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, bold)]
        if bullets:
            _run(p, "•  ", size, True, MAROON)
        for rr in runs:
            if len(rr) > 2 and rr[2] == "MATH":
                _run(p, rr[0], size, rr[1], color).font.name = "Cambria Math"
            else:
                _run(p, rr[0], size, rr[1], rr[2] if len(rr) > 2 else color)
    return tb


def rect(slide, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.RECTANGLE, lw=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    nostyle(s)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    s.text_frame.text = ""
    return s


def icon(slide, name, x, y, d, circle=MAROON, color="w"):
    """Icon in a filled circle (circle=None: bare icon)."""
    if circle is not None:
        rect(slide, x, y, d, d, fill=circle, shape=MSO_SHAPE.OVAL)
        pad = d * 0.24
        slide.shapes.add_picture(str(ICONS / f"{name}_{color}.png"), Inches(x + pad), Inches(y + pad),
                                 Inches(d - 2 * pad), Inches(d - 2 * pad))
    else:
        slide.shapes.add_picture(str(ICONS / f"{name}_{color}.png"), Inches(x), Inches(y), Inches(d), Inches(d))


def arrow(slide, x1, y1, x2, y2, color=GRAY, w=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    nostyle(c)
    c.line.color.rgb = color
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def card(slide, x, y, w, h, title, body, ic=None, fill=PANEL, title_size=14, body_size=12, accent=MAROON):
    rect(slide, x, y, w, h, fill=fill)
    rect(slide, x, y, w, 0.06, fill=accent)
    tx = x + 0.15
    if ic:
        icon(slide, ic, x + 0.15, y + 0.2, 0.42)
        tx = x + 0.68
    text(slide, tx, y + 0.2, w - (tx - x) - 0.12, 0.45, title, size=title_size, bold=True,
         anchor=MSO_ANCHOR.MIDDLE if ic else MSO_ANCHOR.TOP)
    text(slide, x + 0.15, y + 0.72, w - 0.3, h - 0.8, body, size=body_size, color=INK)


def bignum(slide, x, y, w, num, label, color=MAROON, num_size=34, label_size=12, rule=True):
    if rule:
        rect(slide, x, y, w, 0.04, fill=color)
    text(slide, x, y + 0.08, w, 0.65, num, size=num_size, bold=True, color=color)
    text(slide, x, y + 0.78, w, 0.8, label, size=label_size, color=INK)


def table(slide, x, y, w, rows, widths, size=11, header=True, hi=(), row_h=0.28, align_right=()):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(r):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_head = header and i == 0
            cell.fill.solid()
            cell.fill.fore_color.rgb = MAROON if is_head else (LIGHTGOLD if i in hi else (WHITE if i % 2 else PANEL))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j in align_right else PP_ALIGN.LEFT
            runs = val if isinstance(val, list) else [(str(val), is_head)]
            for rr in runs:
                _run(p, rr[0], size, rr[1] or is_head, WHITE if is_head else (rr[2] if len(rr) > 2 else INK))
    return gf


def bar_chart(slide, x, y, w, h, cats, vals, colors=None, horizontal=True, number_format="0", maxv=None,
              font=11, gap=60, label_color=INK):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("s", vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name, ch.font.size = FONT, Pt(font)
    ch.font.color.rgb = INK
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = number_format, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.bold, dl.font.name = Pt(font), True, FONT
    dl.font.color.rgb = label_color
    va = ch.value_axis
    va.has_major_gridlines = False
    va.visible = False
    if maxv:
        va.maximum_scale = maxv
    va.minimum_scale = 0
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(font)
    ca.format.line.color.rgb = MIDGRAY
    ca.has_major_gridlines = False
    if horizontal:
        ca.reverse_order = True  # first category on top
    ser = plot.series[0]
    for i, v in enumerate(vals):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = (colors[i] if colors else GRAY)
        pt.format.line.fill.background()
    return gf


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def content_slide(title):
    s = prs.slides.add_slide(CONTENT)
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx == 0:
            ph.text_frame.text = title
        else:
            ph._element.getparent().remove(ph._element)
    return s


def cover_slide(title, subtitle, names):
    s = prs.slides.add_slide(COVER)
    for ph in s.placeholders:
        i = ph.placeholder_format.idx
        if i == 0:
            tf = ph.text_frame
            for k, ln in enumerate(title.split("\n")):
                para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                r = para.add_run()
                r.text = ln
                r.font.size = Pt(36)
        elif i == 2:
            ph.left, ph.top, ph.width, ph.height = Emu(338325), Emu(2989575), Inches(6.5), Emu(441600)
            ph.text_frame.text = subtitle
        elif i == 1:
            ph.left, ph.top, ph.width, ph.height = Inches(4.3), Emu(3871400), Inches(5.33), Emu(979500)
            tf = ph.text_frame
            for k, ln in enumerate(names):
                para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                etree.SubElement(para._p.get_or_add_pPr(), qn("a:buNone"))
                para._p.get_or_add_pPr().set("marL", "0")
                para._p.get_or_add_pPr().set("indent", "0")
                rn = para.add_run()
                rn.text = ln
                rn.font.size = Pt(12)
    return s


def source(slide, t, y=5.02):
    text(slide, L, y, 8.6, 0.3, t, size=9, color=GRAY)


def eyebrow(slide, t, y=0.86):
    text(slide, L, y, 9.2, 0.25, t, size=10, bold=True, color=MAROON)


# ------------------------------------------------------------------ slides: AIC progress deck (Ibraheem)
NAMES = ["Arjun S Yadav · Chinmay Prashanth · Muhammad Ibraheem", "Arizona State University · 22 November 2026"]
s = cover_slide("VLA policies and reinforcement\nlearning for robotic cable insertion",
                "EEE 598 · AI for Industry Challenge in Gazebo", NAMES)
notes(s, "Thirty-second summary: we went from SmolVLA to X-VLA, fixed a labelling bug that made policies stall, "
         "built a camera-based port detector, and compared residual RL, DSRL and Recap. Recap gave the clearest gain; "
         "SC ports remain unsolved.\n\nAbstract (from the progress deck): Intrinsic AI for Industry Challenge in Gazebo. "
         "Imitation from 585 scripted demonstrations, camera-based port localization, and three RL methods on a frozen "
         "X-VLA. Best policy: X-VLA + Recap, 1163/3600 over 36 test trials with two full insertions.")

# 2 Task and protocol
s = content_slide("Task and evaluation protocol")
eyebrow(s, "1 · TASK AND PROTOCOL")
for i, (ic, t, b) in enumerate([
        ("plug", "Task", "Insert a held SFP or SC plug into the port named in text, on a randomized board."),
        ("camera", "Observation", "3 wrist RGB cameras (288×256), TCP pose and velocity, wrist force/torque."),
        ("route", "Action", "6-D TCP twist at 10 Hz; 10 of each 30–50-step chunk run, replanned every 1 s.")]):
    y = 1.2 + i * 1.18
    icon(s, ic, L, y, 0.55)
    text(s, L + 0.72, y - 0.02, 3.4, 0.3, t, size=14, bold=True)
    text(s, L + 0.72, y + 0.3, 3.4, 0.8, b, size=12)
table(s, 4.75, 1.2, 4.88, [["Item", "Value"],
                           ["Test set", "36 trials: seeds 100, 101 (held out) + 12 training boards"],
                           ["Score", "AIC engine, max 100 per trial (3600 total)"],
                           ["Insertion", "engine event; partial = within 1 cm, not seated"],
                           ["Trial limit", "60 s simulated"],
                           ["Safety stops", "6 cm/s, 0.4 rad/s, 30 N, 40 cm drift"],
                           ["Expert reference", "CheatCode (ground-truth poses): 1039/1200 on seed 100"]],
      [1.35, 3.53], size=11, row_h=0.42)
text(s, 4.75, 4.32, 4.88, 0.5, "A 12-trial set varies by about ±30–40 points between runs.", size=11, color=GRAY)
notes(s, "UR5e with gripper holds an SFP module or SC fiber plug; the target port is named in text (e.g. sfp_port_0 on "
         "nic_card_mount_3). Board pose, card mounts and port positions are randomized per trial. Cameras are 1152×1024, "
         "used at 288×256. Simulator: official AIC Gazebo stack (ROS 2 Kilted), real time, one RTX 5070 (12 GB). "
         "All scores in this deck use this protocol.")

# 3 Demonstration data
s = content_slide("585 demos, one labelling bug fixed")
eyebrow(s, "2 · DEMONSTRATION DATA")
for i, (n, lab) in enumerate([("731", "CheatCode trials recorded (30 seeds)"), ("585", "inserted, used for training"),
                               ("55%", "of each demo was the arm frozen after seating")]):
    bignum(s, L + i * 1.55, 1.2, 1.4, n, lab, num_size=30, label_size=11)
card(s, L, 2.95, 4.5, 1.9, "Label fix (v2 onward)",
     "Actions came from measured motion, so the ~18 s after seating read “velocity 0” and policies stalled near "
     "the port. Fix: trim to 1 s after rest, label from commanded poses, add wrist force/torque (state 26 → 32).",
     ic="screwdriverwrench", body_size=11)
table(s, 5.1, 1.2, 4.53, [["Set", "Demos", "Frames", "Content"],
                          ["v1", "256", "77,453", "measured-motion actions"],
                          ["v2", "256", "41,423", "trimmed, commanded, F/T"],
                          ["v3", "427", "69,367", "v2 fixes, more boards"],
                          ["v4", "585", "94,945", "v2 fixes; X-VLA 19-D state"],
                          ["Recap r1", "585+100", "143,847", "demos + rollouts, tagged"],
                          ["Recap r2", "585+360", "≈ 270k", "two rounds of rollouts"]],
      [0.85, 0.8, 0.85, 2.03], size=10, row_h=0.36, align_right=(1, 2))
text(s, 5.1, 3.85, 4.53, 0.9, "LeRobot v3.0 datasets, public on Hugging Face. Rollouts carry per-second "
     "ground-truth distance and reward.", size=10, color=GRAY)
notes(s, "Actions were derived from measured TCP motion, so the ~18 s after the plug seated read as velocity 0: "
         "policies learned to stall near the port. Fix: trim to 1 s after rest, label from CheatCode's commanded poses, "
         "append wrist force/torque (state 26 to 32). v1: measured-motion actions, 26-D state. v2: tail trimmed, commanded "
         "actions, F/T. v3: v2 fixes, more boards. v4: v2 fixes; the X-VLA variant uses a 19-D state. Recap r1: demos + "
         "X-VLA rollouts, advantage-tagged. Recap r2: demos + rollouts from two rounds.")

# 4 Imitation learning
s = content_slide("X-VLA leads the imitation baselines")
eyebrow(s, "3 · IMITATION LEARNING")
table(s, L, 1.2, 5.35, [["Policy (data, steps)", "Seed 100", "36 trials", "Insertions"],
                        ["SmolVLA 450M (v1, 15k)", "351", "–", "0"],
                        ["SmolVLA + context (v1, 15k)", "337", "–", "0"],
                        ["SmolVLA (v3, 40k)", "272", "722", "1 partial"],
                        ["ACT 52M, scratch (v3, 50k)", "268", "–", "0"],
                        [[("X-VLA 0.9B, full (v4, 60k)", True)], "327", "1052", "2 full, 1 partial"]],
      [2.35, 0.85, 0.85, 1.3], size=11, row_h=0.4, hi=(5,), align_right=(1, 2))
text(s, L, 3.75, 5.35, 0.8, "SmolVLA trains only its 100M action expert (vision frozen); X-VLA trains all 879M "
     "parameters, Florence-2 included.", size=10, color=GRAY)
bignum(s, 6.1, 1.2, 3.5, "722 → 1052", "36-trial score, SmolVLA v3 → X-VLA", num_size=30)
bignum(s, 6.1, 2.75, 3.5, "2 full", "first full insertions, all on SFP ports, once the vision encoder trains",
       num_size=30)
notes(s, "Observations: the label fix removed early stalling but exposed aiming: SmolVLA v2/v3 pushed in the wrong place "
         "(4 force stops) and SFP depth was confused with SC depth. SmolVLA failed equally on its own training boards "
         "(235 vs 272), so the gap is not generalization. Plain-language context and a target image gave no gain with a "
         "frozen vision encoder. Training the vision encoder (X-VLA) gave the first full insertions, all on SFP ports.")

# 5 Port localization
s = content_slide("Wrist cameras locate the port to ~1 mm")
eyebrow(s, "4 · PORT LOCALIZATION")
steps = [("tag", "Free labels", "Recorded poses projected into 531k images"),
         ("eye", "Detector", "ResNet-18 heatmap, told which port to find"),
         ("crosshairs", "Triangulation", "3 cameras, 19 cm baseline, running median"),
         ("bullseye", "Port − TCP", "3-D target from images only")]
for i, (ic, t, b) in enumerate(steps):
    x = L + i * 2.35
    rect(s, x, 1.2, 2.0, 1.45, fill=LIGHTGOLD if i == 3 else PANEL)
    icon(s, ic, x + 0.12, 1.32, 0.42)
    text(s, x + 0.62, 1.32, 1.35, 0.42, t, size=12, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.12, 1.85, 1.78, 0.75, b, size=10)
    if i < 3:
        arrow(s, x + 2.03, 1.92, x + 2.32, 1.92)
table(s, L, 2.95, 4.9, [["Held-out boards (seeds 26–30)", "Value"],
                        ["Recall when visible / false positives", "76% / 6%"],
                        ["Pixel error, median / p90", "0.56 / 1.5 px"],
                        ["3-D error, median / p90", "1.1 / 4.5 mm"],
                        ["Frames under 2 mm / under 5 mm", "74% / 91%"]],
      [3.4, 1.5], size=11, row_h=0.34, hi=(3,), align_right=(1,))
card(s, 5.55, 2.95, 4.08, 1.75, "As a policy input it hurt",
     "X-VLA with the estimate in its state: 896 vs 1052. About 1 frame in 6 on unseen boards locks onto a look-alike "
     "port, and the policy follows it.", ic="triangleexclamation", body_size=11)
notes(s, "Free labels: recorded port poses + TCP pose + camera mounts projected into every frame: 531k images, 0.0 px "
         "from verified projections. Detector: ResNet-18 heatmap, conditioned on target module and slot (look-alike cards), "
         "6 epochs. Triangulation: least-squares ray intersection across 3 cameras (19 cm baseline), causal running median "
         "in base frame. As a policy input it hurt: X-VLA fine-tuned with the estimate in its state scored 896 vs 1052 "
         "(1 partial insertion). On unseen boards about 1 frame in 6 locks onto a look-alike port (p90 error 45 mm), and "
         "the policy follows it confidently. Usable only with stricter multi-camera agreement or as an input RL can learn "
         "to discount.")

# 6 RL setup
s = content_slide("Three RL methods on a frozen X-VLA")
eyebrow(s, "5 · REINFORCEMENT-LEARNING SETUP")
rect(s, L, 1.2, 9.26, 0.62, fill=PANEL)
icon(s, "gaugehigh", L + 0.1, 1.26, 0.5)
text(s, L + 0.75, 1.2, 8.4, 0.62, [[("Reward per 1 s step: ", True), ("100·Δd (cm of progress) − 0.5 if |ΔF| > 20 N; "
     "+50 on insertion; −10 on safety stop. Ground truth only in reward and critic.", False)]], size=12,
     anchor=MSO_ANCHOR.MIDDLE)
for i, (ic, t, b, d) in enumerate([
        ("sliders", "Residual SAC", "A bounded twist (±2 cm/s) is added to X-VLA's chunk.", "280 online episodes"),
        ("dice", "DSRL", "SAC picks X-VLA's flow-matching start noise (30×20, ±1.5σ).", "300 online episodes"),
        ("medal", "Recap", "X-VLA itself is fine-tuned, conditioned on “Advantage: positive”.",
         "demos + recorded rollouts, offline")]):
    x = L + i * 3.13
    card(s, x, 2.05, 2.95, 2.0, t, [b, [("Data: ", True), (d, False)]], ic=ic, body_size=11,
         fill=LIGHTGOLD if i == 2 else PANEL)
text(s, L, 4.2, 9.26, 0.6, "AIC Gazebo with ground truth on; ~55 episodes/h per simulator; 2 parallel simulators "
     "on 12 GB. Recap follows Physical Intelligence's π*0.6.", size=11, color=GRAY)
notes(s, "Environment and reward: AIC Gazebo with ground truth on; one RL step = one 1 s chunk; ~55 episodes/h per "
         "simulator. Reward per step: 100·Δd (cm of plug-to-port progress, potential-based) − 0.5 if |ΔF| > 20 N; +50 on "
         "insertion; −10 on safety stop. Ground truth enters only the reward and the critic (asymmetric actor-critic); "
         "policies see cameras and proprioception. Parallel simulators: per-environment Zenoh port, ROS domain and Gazebo "
         "partition in one container (2 on 12 GB). Recap: cross-fitted privileged value function, per-chunk advantage = "
         "discounted return − V, top 30% + insertion chunks positive.")

# 7 Results (native bar chart)
s = content_slide("Results over the 36 test trials")
eyebrow(s, "6 · RESULTS")
cats = ["SmolVLA v3", "X-VLA + port-estimate input", "X-VLA + Recap round 2", "X-VLA + residual SAC", "X-VLA",
        "X-VLA + DSRL", "X-VLA + Recap round 1"]
vals = [722, 896, 949, 1011, 1052, 1077, 1163]
bar_chart(s, L, 1.15, 6.2, 3.7, cats, vals, colors=[GRAY] * 6 + [MAROON], maxv=1300, font=11)
card(s, 6.8, 1.2, 2.83, 3.55, "Insertions",
     ["Full / partial", "Recap r1: 2 / 0", "DSRL: 1 / 2", "X-VLA: 2 / 1", "Residual: 1 / 0", "Recap r2: 0 / 0", "Port input: 0 / 1",
      "SmolVLA v3: 0 / 1"], ic="flagcheckered", body_size=11)
source(s, "Total AIC score, max 3600. All Recap and RL rows start from the same fine-tuned X-VLA. "
          "Recap r1 per set: 368 / 348 / 447.")
notes(s, "Recap round 1 is the best policy; online RL stays within noise (title of the progress deck's results slide). "
         "Total AIC score over the 36 test trials (max 3600). Full / partial insertions: Recap r1 2/0, DSRL 1/2, X-VLA "
         "2/1, residual 1/0, Recap r2 0/0, port input 0/1, SmolVLA v3 0/1. Recap r1 per set: 368 / 348 / 447. "
         "Note: Chinmay's seeded re-runs (slides 13 and 16 of his part) do not reproduce the Recap gain.")

# 8 Recap iterations
s = content_slide("Recap round 2 fell below round 1")
eyebrow(s, "7 · RECAP ITERATIONS")
table(s, L, 1.2, 5.6, [["X-VLA + Recap", "Round 1", "Round 2"],
                       ["Rollouts (usable)", "120 (100)", "380 (360)"],
                       ["Collection", "1 simulator", "2 parallel"],
                       ["Insertion chunks in rollouts", "4", "15"],
                       ["Value fn vs return, correlation", "0.20", "0.55"],
                       ["Rollout : demo frames", "≈ 0.5 : 1", "≈ 2 : 1"],
                       ["Fine-tune steps (start)", "20k (X-VLA)", "40k (round 1)"],
                       [[("Score, 36 trials", True)], [("1163", True)], "949"],
                       ["Insertions", "2 full", "0"]],
      [2.7, 1.45, 1.45], size=11, row_h=0.36, hi=(7,), align_right=(1, 2))
bignum(s, 6.3, 1.2, 3.3, "1163 → 949", "36-trial score, round 1 → round 2", num_size=30)
card(s, 6.3, 2.75, 3.33, 2.0, "Likely cause",
     "Rollouts outnumber demos and “positive” is the best 30% of mediocre behaviour. Overtraining at 40k steps "
     "is not ruled out.", ic="magnifyingglass", body_size=11)
notes(s, "Positive threshold fixed at the top 30% of chunks in both rounds. Round 1 mostly removed bad trials (5 under 15 "
         "points vs 7 for X-VLA) and ended more trials 2–5 cm from the port; insertion count did not rise. In round 2, "
         "rollouts outnumber demos and 'positive' is the best 30% of mediocre behaviour, so conditioning on it pulls toward "
         "the policy's own average. Overtraining at 40k steps is not ruled out; the 20k checkpoint has not been evaluated.")

# 9 Failure analysis (native bar chart of rates)
s = content_slide("Success depends on port and mount")
eyebrow(s, "8 · FAILURE ANALYSIS")
fcats = ["SFP ports", "SC ports (partial only)", "nic_card_mount_0", "nic_card_mount_1", "nic_card_mount_4",
         "nic_card_mount_2", "nic_card_mount_3", "sc_port_1"]
fvals = [3.1, 0.6, 5.1, 5.1, 2.7, 1.7, 0.9, 0.0]
bar_chart(s, L, 1.15, 5.3, 3.65, fcats, fvals, colors=[MAROON, GRAY, MAROON, MAROON, GRAY, GRAY, GRAY, GRAY],
          number_format='0.0"%"', maxv=6.5, font=11)
for i, (ic, t) in enumerate([
        ("xmark", "No full SC insertion yet; SC trials also start farther away."),
        ("rotate", "Board rotation and position: no measurable effect."),
        ("rulerhorizontal", "Most failures end 4–6 cm from the port."),
        ("dice", "Noise moves outcomes 0.5 cm (SmolVLA), 1.4 cm (X-VLA): little for DSRL.")]):
    y = 1.2 + i * 0.9
    icon(s, ic, 5.9, y, 0.45)
    text(s, 6.5, y - 0.02, 3.13, 0.8, t, size=11, anchor=MSO_ANCHOR.TOP)
source(s, "Success rate per target, 976 X-VLA-family trials (evaluations, RL, collection): 18 full, 4 partial insertions.")
notes(s, "Success rates: SFP ports 20/639 (3.1%); SC ports 2/337 (0.6%, partial only); nic_card_mount_0 7/137 (5.1%); "
         "nic_card_mount_1 6/118 (5.1%); nic_card_mount_4 4/148 (2.7%); nic_card_mount_2 2/121 (1.7%); nic_card_mount_3 "
         "1/115 (0.9%); sc_port_1 0/151 (0%). No policy has fully inserted an SC plug. SC trials start farther away (266 of "
         "275 starts beyond 28 cm are SC), so port type and start distance are confounded. Board rotation and position have "
         "no measurable effect (1.3% vs 1.6% across yaw bins; identical x, y ranges). Most failures end 4–6 cm from the "
         "port with the approach direction largely correct (SmolVLA v3: cosine 0.98 to CheatCode's); misses are depth and "
         "the last centimetres. Flow-matching noise moves the outcome little on a fixed board (spread 0.5 cm SmolVLA, "
         "1.4 cm X-VLA), which bounds what DSRL can steer.")

# 10 Limitations and next steps
s = content_slide("Open problems and next steps")
eyebrow(s, "9 · LIMITATIONS AND NEXT STEPS")
text(s, L, 1.15, 4.4, 0.3, "Limitations", size=14, bold=True, color=MAROON)
for i, (ic, t) in enumerate([("chartcolumn", "36 trials per policy: 0–3 insertions cannot separate methods."),
                             ("clock", "Real-time Gazebo limits RL to ~300 episodes per run."),
                             ("brain", "Recap's value function sees privileged state only (corr. 0.20–0.55)."),
                             ("robot", "Simulation only; no transfer to the physical setup yet.")]):
    y = 1.55 + i * 0.8
    icon(s, ic, L, y, 0.42, circle=GRAY)
    text(s, L + 0.55, y, 3.85, 0.7, t, size=11)
text(s, 5.2, 1.15, 4.4, 0.3, "Next", size=14, bold=True, color=MAROON)
for i, (ic, t) in enumerate([("medal", "Recap round 3: demos as majority, top 10–15% positive, image value fn."),
                             ("chartcolumn", "Larger evaluation (~60 boards), success rate by port type."),
                             ("plug", "SC-targeted demos and evaluation; per-mount analysis."),
                             ("server", "4+ parallel simulators on a 24 GB GPU (remote script ready)."),
                             ("crosshairs", "Port estimate with three-camera agreement as RL input.")]):
    y = 1.55 + i * 0.66
    icon(s, ic, 5.2, y, 0.42)
    text(s, 5.75, y, 3.88, 0.62, t, size=11)
notes(s, "Limitations: 36 trials per policy: insertion counts of 0–3 cannot separate methods; scores carry ±30–40 noise per "
         "12-trial set. Real-time Gazebo limits RL to ~300 episodes per run; online methods were sample-starved. Recap's "
         "value function sees privileged state only (correlation 0.20–0.55 with returns). Simulation only; no transfer to "
         "the physical setup yet. Next: Recap round 3 with demos as the majority, stricter positive threshold (top 10–15%), "
         "image-conditioned value function; evaluate round 2 at 20k steps. Larger evaluation (~60 boards per policy) "
         "reporting success rate by port type. SC-targeted demos and evaluation; per-mount analysis. Scale collection: 4+ "
         "parallel simulators on a 24 GB GPU (remote script ready). Port estimate with three-camera agreement as an "
         "RL-actor input.")

# 11 Artifacts
s = content_slide("Code, data and models are public")
eyebrow(s, "ARTIFACTS")
for i, (ic, t, b) in enumerate([
        ("medal", "Best model", "huggingface.co/muhammad-ibraheem/xvla-recap-v1"),
        ("boxarchive", "Collection (21 items)", "Demo sets v1–v4, Recap sets, 380 raw rollouts with rewards, all "
                                                "checkpoints, port detector, RL agents"),
        ("github", "Code", "github.com/ibraheem-111/eee598-aic, with videos of every evaluated trial"),
        ("code", "Key scripts", "convert_demos.py · train_xvla.sh, train_act.sh, train_smolvla.sh · rl_train.sh, "
                                "rl_parallel.sh · build_recap_dataset.py · train_port_detector.py, port_estimator.py")]):
    x, y = L + (i % 2) * 4.7, 1.2 + (i // 2) * 1.85
    card(s, x, y, 4.55, 1.7, t, b, ic=ic, body_size=11)
notes(s, "Released: best model huggingface.co/muhammad-ibraheem/xvla-recap-v1. Collection (21 items): demo sets v1–v4, "
         "Recap sets, 380 raw rollouts with rewards, all checkpoints, port detector, RL agents. Code: "
         "github.com/ibraheem-111/eee598-aic, with third-person videos of every evaluated trial. Key scripts: "
         "convert_demos.py (trim, commanded actions, force/torque); train_xvla.sh, train_act.sh, train_smolvla.sh; "
         "rl_train.sh, rl_parallel.sh (residual SAC, DSRL, coverage, parallel simulators); build_recap_dataset.py (value "
         "function, advantages, tagged dataset); train_port_detector.py, port_estimator.py.")

# ------------------------------------------------------------------ Chinmay, slides 9-16
s = content_slide("X-VLA misses sideways, not short")
eyebrow(s, "WHERE IT FAILS · COURSE: L12 REWARD DESIGN")
for i, (n, lab) in enumerate([("32 mm", "median final sideways offset of the base policy to the target port"),
                               ("23 mm", "spacing between neighbouring SFP ports: a miss this size lands in the wrong one"),
                               ("5 mm", "sideways tolerance the scorer allows for a partial insertion")]):
    bignum(s, L + i * 3.13, 1.25, 2.9, n, lab, color=MAROON if i == 0 else INK, num_size=40, label_size=12)
rect(s, L, 3.3, 9.26, 0.85, fill=PANEL)
icon(s, "arrowsleftright", L + 0.15, 3.42, 0.6)
text(s, L + 0.95, 3.3, 8.2, 0.85, "Across 380 logged rollouts the plug reaches the port's depth but ends 2–4 cm off to the "
     "side. The open problem is lateral alignment.", size=13, anchor=MSO_ANCHOR.MIDDLE)
source(s, "Base policy xvla-aic-v4, n = 36 eval episodes (95% CI of median 26.7–41.3 mm); 380 rollouts from Ibraheem's "
          "Recap runs")
notes(s, "Chinmay, slide 9. Where X-VLA fails: the base policy's median final sideways offset to the target port is 32 mm, "
         "neighbouring SFP ports are 23 mm apart, and the scorer allows 5 mm sideways for a partial insertion. Across 380 "
         "logged rollouts the plug reaches the port's depth but ends 2–4 cm off to the side: the open problem is lateral "
         "alignment. Source: base policy xvla-aic-v4, n = 36 eval episodes (95% CI of median 26.7–41.3 mm).")

s = content_slide("The RL problem in course notation")
eyebrow(s, "THE RL PROBLEM · COURSE: L2/L3 MDP · L15 ACTOR-CRITIC · L16 PRIVILEGED CRITIC · L13")
table(s, L, 1.15, 5.95, [["DSRL", "In our system"],
                         ["State Sₜ", "actor: joints, TCP pose/velocity, wrist F/T, camera port estimate, port type, time "
                                      "(45-dim). Critic also: privileged sim state, plug and port pose (7), like ANYmal's teacher (L16)"],
                         ["Action Aₜ", "noise w ∈ [−1.5, 1.5]^(30×20 = 600), chosen once per chunk"],
                         ["Transition", "frozen X-VLA (3 wrist cameras) decodes w into a 1 s chunk, run in Gazebo"],
                         ["Reward Rₜ₊₁", "per chunk: his 100·Δd + 50 on insertion − penalties, or ours (scorer potentials)"],
                         ["Return Gₜ", "Σₖ γᵏ Rₜ₊ₖ₊₁, γ = 0.97 per chunk; ≈ 55 chunks per 60 s; time-out bootstraps"],
                         ["Agent (L15)", "actor π_θ(w | Sₜ), tanh-squashed Gaussian; critics Q_φ1, Q_φ2 + target nets; "
                                         "replay buffer (off-policy)"]],
      [1.15, 4.8], size=9, row_h=0.42, hi=(2,))
rect(s, 6.55, 1.15, 3.08, 3.55, fill=PANEL)
icon(s, "medal", 6.7, 1.3, 0.45)
text(s, 7.25, 1.3, 2.3, 0.45, "Recap, same notation", size=12, bold=True, color=MAROON, anchor=MSO_ANCHOR.MIDDLE)
text(s, 6.7, 1.9, 2.8, 2.75, ["Gₜ with γ = 0.9 per chunk; V(Sₜ) fitted on rollouts (5-fold cross-fit).",
                              [("Aₜ = Gₜ − V(Sₜ)", True), ("; top 30% of chunks labelled “positive”.", False)],
                              [("Fine-tune X-VLA on the label, prompt “positive” at test: raises ln π for high-advantage "
                               "actions, like ", False), ("ĝ ∝ Σ Aₜ ∇ ln π_θ(Aₜ|Sₜ)", False, "MATH"), (" (L13/L15).", False)]], size=10, spacing=8)
source(s, "eee598-aic @ e83ad59: rl_agent.py (γ = 0.97), server.py (actor features, privileged critic input), "
          "build_recap_dataset.py (γ = 0.9, top 30%)")
notes(s, "Chinmay, slide 10. The DSRL and Recap problems written in course notation (L2/L3 MDP, L15 actor-critic, L16 "
         "privileged critic, L13 policy gradient). Source: eee598-aic @ e83ad59.")

s = content_slide("How every number here is measured")
eyebrow(s, "METHOD · COURSE: L8 REPORTING RL RESULTS")
for i, (ic, k, t, b) in enumerate([
        ("dice", "SEEDS · L8", "Independent runs", "Fresh agent per seed: 6 per DSRL reward; 3 for Recap and entropy runs."),
        ("flask", "FINAL POLICY · L8", "Evaluated separately", "Frozen, deterministic; 12 boards × 3 runs = 36 episodes "
                                                                "per seed."),
        ("scalebalanced", "UNCERTAINTY · L8", "CI, not ±1 sd", "95% bootstrap CI, paired by board.")]):
    x = L + i * 3.13
    rect(s, x, 1.2, 2.95, 2.35, fill=PANEL)
    icon(s, ic, x + 0.18, 1.35, 0.5)
    text(s, x + 0.18, 1.95, 2.6, 0.25, k, size=9, bold=True, color=MAROON)
    text(s, x + 0.18, 2.2, 2.6, 0.4, t, size=15, bold=True)
    text(s, x + 0.18, 2.65, 2.6, 0.85, b, size=11)
text(s, L, 3.8, 9.26, 0.4, "Rules written before each run; every seed reported; base re-run in each batch.", size=13)
source(s, "Official scorer, limits 60 s, 6 cm/s, 30 N; 10,000 reps; ±1 sd only on learning curves (not a CI); "
          "dsrl/paired.py; pre-registrations in recap_fixed/, dsrl/, recap_repro/, docs/STATUS.md", y=4.85)
notes(s, "Chinmay, slide 11. How every number in his part is measured (course L8): independent runs with a fresh agent per "
         "seed, the final policy evaluated separately (frozen, deterministic, 36 episodes per seed), and 95% bootstrap "
         "confidence intervals paired by board. Rules written before each run; every seed reported; base re-run in each batch.")

s = content_slide("The reward paid +50 for the wrong port")
eyebrow(s, "REWARD BUG AND FIX · COURSE: L12 REWARD DESIGN (GT SOPHY)")
table(s, L, 1.2, 6.2, [["Per episode", "Official scorer", "His RL reward", "Our fix"],
                       ["Correct-port insertion", "+75", "+50", "+75"],
                       ["Wrong-port insertion", "−12", "+50", "−12"],
                       ["1 cm less sideways error, near the port", "—", "0.44 points", "4.8 points"],
                       ["Return vs official score (216 episodes)", "—", "Spearman 0.35", "Spearman 0.86"]],
      [2.6, 1.2, 1.2, 1.2], size=11, row_h=0.45, hi=(2,))
bignum(s, 6.95, 1.2, 2.68, "10 of 15", "logged insertions went into the wrong port, each earning +50", num_size=36)
text(s, L, 3.75, 9.26, 0.6, "Fix: reward = change in the scorer's own terms, plus a sideways-alignment term.", size=13)
source(s, "Scorer terms from ScoringTier2.cc; 380 logged rollouts; reward_analysis/REPORT.md §4 "
          "(reward_features.chunk_reward)")
notes(s, "Chinmay, slide 12. The RL reward used in Ibraheem's runs paid +50 for any insertion event, including the wrong "
         "port, which the official scorer penalizes (−12). 10 of 15 logged insertions went into the wrong port. The fix "
         "uses the change in the scorer's own terms plus a sideways-alignment term; its return correlates with the official "
         "score at Spearman 0.86 vs 0.35.")

s = content_slide("No RL method beat the imitation base")
eyebrow(s, "RESULTS · PRE-REGISTERED · 12 BOARDS × 3 RUNS, BASE RE-RUN IN EACH BATCH · COURSE: L8")
table(s, L, 1.15, 9.26, [["Policy", "n (seeds)", "Mean (base)", "− base, paired 95% CI", "Correct port", "Contact"],
                         ["Base: X-VLA v4 (BC), 4 batches", "144", "29.4–30.9", "—", "3/144 = 2.1%", "1/144"],
                         ["Recap v1 retrained (his recipe)", "108 (3)", "28.3 (29.9)", "−1.6 [−5.6, 2.2]", "0/108 = 0.0%", "0/108"],
                         ["DSRL ±1.5, his reward", "216 (6)", "22.8 (29.4)", "−6.6 [−12.5, −0.6]", "0/216 = 0.0%", "2/216"],
                         ["DSRL ±1.5, recommended reward", "216 (6)", "28.7 (30.9)", "−2.2 [−6.2, 1.6]", "8/216 = 3.7%", "2/216"],
                         ["DSRL ±1.5, rec. reward, paper entropy", "108 (3)", "27.4 (29.5)", "−2.1 [−7.9, 4.4]", "4/108 = 3.7%", "0/108"]],
      [2.9, 0.9, 1.15, 1.75, 1.45, 1.11], size=10, row_h=0.32, hi=(4,))
for i, (ok, t, b) in enumerate([(False, "1 · Success rate", "Not met: 3.7% vs 2.1%, too few to claim"),
                                (False, "2 · Mean score", "Not met: no interval above 0"),
                                (True, "3 · Penalties", "Met: contact in ≤ 1% everywhere"),
                                (False, "4 · Over BC", "Not met; reward fix +5.5 [1.6, 9.9] over his")]):
    x = L + i * 2.33
    rect(s, x, 3.25, 2.2, 1.4, fill=PANEL)
    icon(s, "check" if ok else "xmark", x + 0.12, 3.37, 0.38, circle=MAROON if ok else GRAY)
    text(s, x + 0.58, 3.37, 1.55, 0.38, t, size=11, bold=True, color=MAROON, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.12, 3.85, 1.98, 0.8, b, size=10)
source(s, "n = episodes; base = v4 in the same batch; board bootstrap, 10,000 reps; contact = scorer's −24 off-limit term; "
          "reward_analysis/results/final_criteria.csv, summary_methods.csv", y=4.8)
notes(s, "Chinmay, slide 13 (original title: No RL method beat the imitation baseline). Pre-registered comparison, 12 boards "
         "× 3 runs, base re-run in each batch. Criteria: 1 success rate not met (3.7% vs 2.1%, too few to claim); 2 mean "
         "score not met (no interval above 0); 3 penalties met (contact in ≤ 1% everywhere); 4 over BC not met (reward fix "
         "+5.5 [1.6, 9.9] over his).")

s = content_slide("Base vs DSRL: an insertion")
eyebrow(s, "IN THE SIMULATOR · EVAL BOARD 6 (SFP), ONE RECORDING EACH, REAL TIME · COURSE: L8 (ONE RUN ≠ EVALUATION)")
img_w = 9.26
s.shapes.add_picture(str(SRC / "dsrl_vs_base.jpg"), Inches(L), Inches(1.15), Inches(img_w), Inches(img_w * 516 / 1664))
text(s, L, 4.15, 9.26, 0.6, [[("Left: base, no insertion, 36.7. Right: DSRL seed 2, correct port at 28.5 s, 93.2. ", False),
                             ("Chosen clips", True), (": DSRL inserted in 2 of 3 recordings, the base correctly in 0 of 4.",
                                                     False)]], size=12)
source(s, "Recorded for video, not scored eval episodes. In the 36-episode eval on this board: seed 2 inserted 1 of 3, base "
          "0 of 3. Overall seed 2 30.6 vs base 29.5 (n = 36 each). runs/videos/board6*", y=4.8)
notes(s, "Chinmay, slide 14 (original title: Base vs DSRL, recommended reward: an insertion). Screenshot from his recordings "
         "of eval board 6. These are chosen clips recorded for video, not scored evaluation episodes; one run is not an "
         "evaluation (L8).")

s = content_slide("Too little steering, diverging critic")
eyebrow(s, "WHY DSRL DIDN'T HELP · COURSE: L10 DEADLY TRIAD · L15 SAC")
for i, (n, lab) in enumerate([("~2 mm", "sideways steering per chunk at ±1.5; misses are 20–40 mm"),
                               ("10⁶–10⁷", "critic Q in the saved agents; reward alone gives 4–20"),
                               ("7 × 10/10", "SAC variants that still blow up (toy ablation, 10 seeds each)")]):
    bignum(s, L + i * 3.13, 1.25, 2.9, n, lab, color=MAROON if i == 1 else INK, num_size=38, label_size=12)
rect(s, L, 3.3, 9.26, 0.85, fill=PANEL)
icon(s, "triangleexclamation", L + 0.15, 3.42, 0.6)
text(s, L + 0.95, 3.3, 8.2, 0.85, "No single SAC fix stops the blow-up; removing the entropy term from the critic target "
     "only delays it.", size=13, anchor=MSO_ANCHOR.MIDDLE)
source(s, "noise_sweep/noise_sweep.py; dsrl/sac_health.py, sac_rounds.py bellman; dsrl/ablation/ (pre-registered); "
          "details in backup")
notes(s, "Chinmay, slide 15 (original title: Too little steering, and a critic that diverges). Why DSRL didn't help: at "
         "±1.5 the noise steers the hand only ~2 mm sideways per chunk while misses are 20–40 mm, and the SAC critic "
         "diverges (Q values of 10^6–10^7 where the reward alone gives 4–20), the deadly triad of L10.")

s = content_slide("Three lessons")
eyebrow(s, "CONCLUSIONS · COURSE: L8 · L12 · L15")
for i, (ic, k, t, b) in enumerate([
        ("dice", "1 · L8", "Evaluate with seeds before believing a gain", "+8.5 became +1.9 on re-run."),
        ("scalebalanced", "2 · L12", "Check the reward against the scorer", "The wrong port earned +50."),
        ("brain", "3 · L15", "DSRL needs a noise-sensitive base", "Scripted demos are the bottleneck.")]):
    x = L + i * 3.13
    rect(s, x, 1.2, 2.95, 2.45, fill=PANEL)
    icon(s, ic, x + 0.18, 1.35, 0.5)
    text(s, x + 0.18, 1.95, 2.6, 0.25, k, size=9, bold=True, color=MAROON)
    text(s, x + 0.18, 2.2, 2.6, 0.8, t, size=14, bold=True)
    text(s, x + 0.18, 3.05, 2.6, 0.55, b, size=11)
text(s, L, 3.9, 9.26, 0.4, [[("Future work: ", True), ("TD3 (no entropy term), more varied demos, more seeds (L8: 10).",
                                                     False)]], size=13)
source(s, "github.com/Chinmay-Prashanth/eee598-aic-dsrl (private) · ibraheem-111/eee598-aic")
notes(s, "Chinmay, slide 16. Three lessons: evaluate with seeds before believing a gain (+8.5 became +1.9 on re-run); check "
         "the reward against the scorer (the wrong port earned +50); DSRL needs a noise-sensitive base (scripted demos are "
         "the bottleneck). Future work: TD3 (no entropy term), more varied demos, more seeds (L8: 10).")

# ------------------------------------------------------------------ control slides
ctl = Presentation(SRC / "control.pptx")
CNOTES = [sl.notes_slide.notes_text_frame.text for sl in ctl.slides]


def tag(slide, x, y, label, fill=GOLD, color=INK, w=None):
    w = w or (0.09 * len(label) + 0.2)
    b = rect(slide, x, y, w, 0.22, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(slide, x, y, w, 0.22, label, size=8, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return b


s = content_slide("From intent to muscle, VLA to motor")
eyebrow(s, "1 / 4 · THE CASCADE")
text(s, L, 1.1, 4.3, 0.25, [[("HUMAN  ", True, MAROON), ("from intent to muscle", False, GRAY)]], size=10)
text(s, 5.0, 1.1, 4.3, 0.25, [[("ROBOT  ", True, MAROON), ("from VLA to motor", False, GRAY)]], size=10)
human = [("Goal and plan", "> 100 ms", "cortex plans, basal ganglia choose"),
         ("Command", "50–100 ms", "motor cortex: trajectory and force"),
         ("Spinal cord", "20–50 ms", "muscle drive, stiffness, reflex gain"),
         ("Muscle", "0 ms", "a spring that stiffens as it works")]
for i, (t, ms, b) in enumerate(human):
    y = 1.4 + i * 0.82
    rect(s, 1.75, y, 2.85, 0.7, fill=PANEL)
    text(s, 1.85, y + 0.06, 1.8, 0.28, t, size=11, bold=True)
    text(s, 3.45, y + 0.06, 1.1, 0.28, ms, size=11, bold=True, color=MAROON, align=PP_ALIGN.RIGHT)
    text(s, 1.85, y + 0.36, 2.7, 0.32, b, size=9)
    if i < 3:
        arrow(s, 3.17, y + 0.7, 3.17, y + 0.82, color=MAROON, w=1.25)
rect(s, L, 1.4, 1.25, 3.16, fill=None, line=MAROON, lw=1.0)
text(s, L + 0.08, 1.48, 1.1, 0.5, "Cerebellum", size=11, bold=True)
text(s, L + 0.08, 1.75, 1.1, 0.7, "predicts what the body will feel", size=9)
text(s, L + 0.08, 2.55, 1.1, 0.5, "ahead of the feedback", size=9, bold=True, color=MAROON)
text(s, L + 0.08, 3.75, 1.1, 0.7, "tunes the next command", size=9, bold=True, color=MAROON)
robot = [("VLA", "160–660 ms", "π0.5 · AIRoA-0, the one we run. Plan the subtask; 1 s of hand poses, twice a second. "
                                "Motion only: no force, no stiffness", "EXISTS", 1.15),
         ("Low level", "5–10 ms", "fixed gains: fast, but blind to the task: where RL goes", "HAND-SET", 0.75),
         ("Drive and gears", "≈ 0", "the actuator", "HARDWARE", 0.55)]
y = 1.4
for i, (t, ms, b, tg, h) in enumerate(robot):
    rect(s, 5.0, y, 2.85, h, fill=PANEL)
    tag(s, 5.08, y + 0.06, tg, fill=GOLD if tg != "HARDWARE" else MIDGRAY)
    text(s, 5.1, y + 0.3, 1.5, 0.28, t, size=11, bold=True)
    text(s, 6.6, y + 0.3, 1.18, 0.28, ms, size=11, bold=True, color=MAROON, align=PP_ALIGN.RIGHT)
    text(s, 5.1, y + 0.58, 2.7, h - 0.6, b, size=9)
    if i < 2:
        arrow(s, 6.42, y + h, 6.42, y + h + 0.1, color=MAROON, w=1.25)
    y += h + 0.1
for i, (t, tg, b, dashed) in enumerate([("World model", "PARTLY", "imagines what the camera will see; π0.7 redraws it "
                                                                  "every 4 s", False),
                                        ("Body model", "MISSING", "predicts what the body will feel, and learns from every "
                                                                  "movement", True)]):
    yy = 1.4 + i * 1.6
    rect(s, 8.05, yy, 1.58, 1.45, fill=None, line=MAROON if dashed else GRAY, lw=1.0)
    tag(s, 8.13, yy + 0.07, tg, fill=MAROON if dashed else GOLD, color=WHITE if dashed else INK)
    text(s, 8.13, yy + 0.34, 1.45, 0.28, t, size=11, bold=True)
    text(s, 8.13, yy + 0.62, 1.45, 0.8, b, size=9)
rect(s, 4.72, 2.35, 0.2, 1.05, fill=GOLD)
text(s, 4.62, 3.42, 0.45, 0.6, "no layer 10–160 ms", size=7, bold=True, color=MAROON, align=PP_ALIGN.CENTER)
text(s, L, 4.68, 9.26, 0.4, "The brain meets a surprise on four clocks at once. A VLA stack like ours has the slowest one, "
     "fixed gains under it, and a predictor that only sees pictures.", size=11, bold=True)
source(s, "Human side after MAE 598 Human Robotics (Prof. H. Lee, ASU), redrawn. Clocks: Pruszynski & Scott 2012; ours, real "
          "arms, 21 Nov 2026.", y=5.1)
notes(s, CNOTES[0])

s = content_slide("One 10 Hz network cannot do it all")
eyebrow(s, "2 / 4 · WHY THE LAYERS ARE SEPARATE")
panels = ["1 · A slip outruns the next look", "2 · Pressing harder hardly shows", "3 · Delay caps the closing speed"]
for i, t in enumerate(panels):
    x = L + i * 3.13
    rect(s, x, 1.15, 2.95, 3.4, fill=PANEL)
    text(s, x + 0.15, 1.22, 2.7, 0.3, t, size=11, bold=True, color=MAROON)
# panel 1: callouts
x = L
text(s, x + 0.15, 1.6, 2.7, 0.25, "HUMAN · the slip starts", size=8, bold=True, color=GRAY)
text(s, x + 0.15, 1.82, 2.7, 0.45, "74 ± 9 ms", size=22, bold=True, color=MAROON)
text(s, x + 0.15, 2.25, 2.7, 0.3, "grip tightens, by reflex", size=10)
text(s, x + 0.15, 2.65, 2.7, 0.25, "ROBOT · OUR VLA", size=8, bold=True, color=GRAY)
text(s, x + 0.15, 2.87, 2.7, 0.45, "160–660 ms", size=22, bold=True, color=INK)
text(s, x + 0.15, 3.3, 2.7, 0.3, "first answer that has seen it", size=10)
text(s, x + 0.15, 3.7, 2.7, 0.8, "Until then the old poses keep playing. A dropped object falls 2.7 cm in 74 ms, "
     "2.1 m in 660 ms.", size=9)
# panel 2: callouts
x = L + 3.13
text(s, x + 0.15, 1.6, 2.7, 0.25, "REAL ARMS · 21 NOV 2026 · UNPLANNED CONTACT", size=8, bold=True, color=GRAY)
text(s, x + 0.15, 1.85, 2.7, 0.45, "1 → 17.9 N m", size=22, bold=True, color=MAROON)
text(s, x + 0.15, 2.28, 2.7, 0.3, "shoulder-pan torque over 3 s of rising push", size=10)
text(s, x + 0.15, 2.7, 2.7, 0.45, "within 21 mm", size=22, bold=True, color=INK)
text(s, x + 0.15, 3.13, 2.7, 0.3, "hand's distance from where it was asked", size=10)
text(s, x + 0.15, 3.55, 2.7, 0.95, "Then it lay on the other arm for 9 s: its lift motor carried 0.2 N m where holding the "
     "arm up takes 9.", size=9)
# panel 3: native chart
x = L + 6.26
text(s, x + 0.15, 1.55, 2.7, 0.3, "v ≤ ΔF / (k × delay)", size=13, bold=True, align=PP_ALIGN.CENTER)
text(s, x + 0.15, 1.85, 2.7, 0.35, "a first touch kept within 0.5 N on a fingertip of 0.5 N/mm", size=8, color=GRAY,
     align=PP_ALIGN.CENTER)
bar_chart(s, x + 0.05, 2.2, 2.85, 1.55, ["10 ms · 200 Hz loop", "100 ms · net at 10 Hz", "156 ms · our VLA"],
          [100, 10, 6], colors=[MAROON, GRAY, GRAY], number_format='0" mm/s"', maxv=140, font=9, gap=40)
text(s, x + 0.15, 3.8, 2.7, 0.7, "It is why our planned descents take their last 15 mm in a full second.", size=9)
text(s, L, 4.62, 9.26, 0.35, "A loop cannot correct what happens faster than its own delay, and it cannot control a force "
     "it never senses.", size=11, bold=True)
source(s, "Real record: our reBot B601-RS arms, 21 Nov 2026. Slip: Johansson & Westling 1987.", y=5.0)
notes(s, CNOTES[1])

s = content_slide("Four parts, four teachers")
eyebrow(s, "3 / 4 · WHAT LEARNS FROM WHAT")
rows = [("Plan", "EXISTS", "the VLA: what to do next", "human demonstrations",
         "12 h on one GPU to post-train; 3.36 billion parameters"),
        ("World model", "PARTLY", "what the camera will see", "video", "offline; 14 billion parameters in π0.7's"),
        ("Body model", "MISSING · OURS: SIM ONLY", "what the body will feel", "its own tracking error, as the cerebellum "
                                                                          "is thought to", "3 s of movement; 19 numbers per arm"),
        ("Low level", "HAND-SET · RL: NEXT SLIDE", "how fast, stiff, hard and tight, every 5 ms",
         "reward for the outcome, as dopamine's prediction error does", "many attempts; a small network")]
for i, (t, tg, sub, src, cost) in enumerate(rows):
    y = 1.3 + i * 0.8
    hl = i in (2, 3)
    rect(s, L, y, 5.6, 0.72, fill=LIGHTGOLD if i == 3 else PANEL, line=MAROON if i == 2 else None, lw=1.0)
    text(s, L + 0.1, y + 0.04, 1.7, 0.25, t, size=11, bold=True)
    text(s, L + 0.1, y + 0.27, 1.7, 0.18, tg, size=7, bold=True, color=MAROON)
    text(s, L + 0.1, y + 0.45, 1.7, 0.3, sub, size=8, color=GRAY)
    text(s, L + 1.85, y + 0.05, 1.95, 0.68, src, size=9, bold=hl)
    text(s, L + 3.85, y + 0.05, 1.7, 0.68, cost, size=8, color=INK)
for hx, hd in ((0.1, "PART OF THE STACK"), (1.85, "LEARNS FROM"), (3.85, "COST")):
    text(s, L + hx, 1.1, 1.9, 0.2, hd, size=8, bold=True, color=GRAY)
text(s, L, 4.57, 5.6, 0.3, "One training signal cannot teach all four.", size=11, bold=True, color=MAROON)
rect(s, 6.15, 1.15, 3.48, 3.4, fill=PANEL)
text(s, 6.28, 1.2, 3.3, 0.28, "What we have of the body model", size=11, bold=True)
text(s, 6.28, 1.45, 3.3, 0.4, "One lift repeated on a simulated two-joint arm: torque feedback still had to add [N m rms]",
     size=8, color=GRAY)
bar_chart(s, 6.2, 1.85, 3.35, 1.45, ["1", "2", "3", "4", "5", "6"], [0.97, 0.40, 0.21, 0.19, 0.19, 0.21],
          colors=[MAROON] + [GRAY] * 5, horizontal=False, number_format="0.00", maxv=1.15, font=8, gap=40)
for i, (tg, t) in enumerate([("SIMULATION", "friction found within 0.07 N m in 3 s"),
                             ("SIMULATION", "a touch felt within 20 ms"),
                             ("REAL ARM", "gravity torque within 5 % of the arm's")]):
    yy = 3.38 + i * 0.37
    tag(s, 6.28, yy, tg, fill=GOLD if tg == "REAL ARM" else MIDGRAY, w=0.95)
    text(s, 7.3, yy - 0.01, 2.28, 0.32, t, size=9)
text(s, L, 4.85, 9.26, 0.35, "Imitation teaches the plan and error teaches the body model. How the hand should behave in "
     "contact has neither in our stack: it is left to reward.", size=10, bold=True)
source(s, "Dopamine: Schultz, Dayan & Montague 1997. Our numbers: simulation unless marked REAL ARM.", y=5.22)
notes(s, CNOTES[2])

s = content_slide("A deep Q-network for hand dynamics")
eyebrow(s, "4 / 4 · REINFORCEMENT LEARNING AT THE LOW LEVEL")
boxes = [("STATE s", "every 5 ms", "what the motors feel: no camera. How far the plan is ahead of the hand; the force "
                                   "the hand exerts; its speed and the axes at their cap; load, resistance, its knobs, the "
                                   "stage. 32 numbers; 27 of them are in our code today", False),
         ("Q-NETWORK Q(s, a)", "NOT BUILT", "the worth of each choice, from here: s → 64 → 64 → one value for each choice. "
                                            "It takes the best one. While learning, sometimes a random one.", True),
         ("ACTION a", "one small step", "nine choices: slower / faster (plan speed), softer / stiffer (stiffness), lower / "
                                        "higher (force cap), looser / tighter (grip), or keep; all inside fixed bounds", False)]
for i, (t, side, b, nb) in enumerate(boxes):
    x = L + i * 3.13
    rect(s, x, 1.15, 2.95, 1.9, fill=LIGHTGOLD if i == 1 else PANEL)
    text(s, x + 0.12, 1.2, 1.75, 0.3, t, size=10, bold=True, color=MAROON)
    if nb:
        tag(s, x + 1.95, 1.22, side, fill=MAROON, color=WHITE, w=0.9)
    else:
        text(s, x + 1.7, 1.2, 1.15, 0.3, side, size=9, bold=True, color=GRAY, align=PP_ALIGN.RIGHT)
    text(s, x + 0.12, 1.5, 2.72, 1.5, b, size=9)
    if i < 2:
        arrow(s, x + 2.97, 2.1, x + 3.11, 2.1, color=MAROON, w=1.5)
rect(s, L, 3.13, 9.26, 0.34, fill=PANEL, line=MIDGRAY)
text(s, L, 3.13, 9.26, 0.34, [[("ENVIRONMENT  ", True, MAROON), ("VLA's poses → torque loop, 200 Hz → arm → contact   "
                                                                  "·   s′, r  5 ms later", False)]], size=9,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
for i, (t, b, nb) in enumerate([
        ("REWARD r", "from what the robot can sense itself: + progress on the task, + the part still held at the end; "
                     "− force over the task's limit, − slip and time", False),
        ("LEARNING (DQN)", "target = r + γ · maxₐ′ Q_target(s′, a′), then move Q(s, a) toward the target. Replay buffer: "
                           "past steps, in random order; Q target: a frozen copy; γ = 0.99: about 0.5 s ahead", False),
        ("TRAINING", "an episode: one grasp or one stroke, a few seconds; first in simulation, then on the arm inside the "
                     "bounds; done on a real arm: a DQN picking stiffness every 20 ms (Oikawa 2020)", True)]):
    x = L + i * 3.13
    rect(s, x, 3.55, 2.95, 1.2, fill=PANEL)
    text(s, x + 0.12, 3.6, 1.8, 0.28, t, size=10, bold=True, color=MAROON)
    if nb:
        tag(s, x + 1.95, 3.62, "NOT BUILT", fill=MAROON, color=WHITE, w=0.9)
    text(s, x + 0.12, 3.88, 2.72, 0.86, b, size=8)
text(s, L, 4.8, 9.26, 0.3, "The VLA says where the hand goes. The Q-network would decide, 200 times a second, how it "
     "behaves on the way: how fast, how stiff, how hard, how tight.", size=10, bold=True)
source(s, "A design for our stack: nothing here is trained or run. DQN: Mnih et al. 2015. Precedent: Oikawa et al. 2020.",
       y=5.17)
notes(s, CNOTES[3])

# ------------------------------------------------------------------ closing
s = cover_slide("Thank you", "Questions and discussion", NAMES)
notes(s, "Closing. Code: github.com/ibraheem-111/eee598-aic. Models and data: huggingface.co/collections/muhammad-ibraheem/"
         "eee-598-aic-cable-insertion.")

# ------------------------------------------------------------------ clean-up: drop template slides, unused layouts
for sldId in ORIG:
    prs.part.drop_rel(sldId.get(qn("r:id")))
    prs.slides._sldIdLst.remove(sldId)
used = {sl.slide_layout.part.partname for sl in prs.slides}
for master in prs.slide_masters:
    lst = master._element.find(qn("p:sldLayoutIdLst"))
    for lid in list(lst):
        rid = lid.get(qn("r:id"))
        if master.part.related_part(rid).partname not in used:
            master.part.drop_rel(rid)
            lst.remove(lid)
prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
