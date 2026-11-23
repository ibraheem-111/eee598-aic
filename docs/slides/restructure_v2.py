#!/usr/bin/env python3
"""In-place edit of the delivered combined deck: unified intro / methods / results / discussion,
Chinmay's revised slides (22 Nov), reconciliations, and video slots. Reuses the drawing helpers of
build_asu_deck.py (its helper section only) on the delivered file; the build script is not re-run."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = Path(sys.argv[1])
OUT_PATH = Path(sys.argv[2])
src = (HERE / "build_asu_deck.py").read_text()
head = src.split("# ------------------------------------------------------------------ slides: AIC progress deck")[0]
head = head.replace('TEMPLATE = HERE / "asu/template.pptx"', f'TEMPLATE = Path({str(DECK)!r})')
g = {"__file__": str(HERE / "build_asu_deck.py"), "__name__": "helpers"}
exec(compile(head, "helpers", "exec"), g)
globals().update({k: v for k, v in g.items() if not k.startswith("__")})

MEDIA = Path("/home/ibraheem/eee598-aic/docs/media")
POST = HERE / "posters"
S = {i + 1: sl for i, sl in enumerate(prs.slides)}  # original numbering 1..24
assert len(S) == 24


# ------------------------------------------------------------------ edit helpers
def frames(slide):
    for sh in slide.shapes:
        if sh.has_text_frame:
            yield sh, sh.text_frame
        if sh.has_table:
            for row in sh.table.rows:
                for c in row.cells:
                    yield sh, c.text_frame


def replace(slide, old, new, count=1):
    """Replace text inside one run (formatting kept). Fails loudly if not found."""
    n = 0
    for _, tf in frames(slide):
        for p in tf.paragraphs:
            for r in p.runs:
                if old in r.text:
                    r.text = r.text.replace(old, new)
                    n += 1
    assert n >= count, (old, n)


def set_para(slide, startswith, new):
    """Replace a whole paragraph (all runs) that starts with `startswith`; first run's format kept."""
    for _, tf in frames(slide):
        for p in tf.paragraphs:
            if "".join(r.text for r in p.runs).startswith(startswith):
                p.runs[0].text = new
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
                return
    raise AssertionError(startswith)


def set_title(slide, t):
    slide.shapes.title.text_frame.paragraphs[0].runs[0].text = t
    for r in slide.shapes.title.text_frame.paragraphs[0].runs[1:]:
        r._r.getparent().remove(r._r)


def set_eyebrow(slide, t):
    for sh in slide.shapes:
        if sh.has_text_frame and abs(sh.top - Inches(0.86)) < Inches(0.02):
            p = sh.text_frame.paragraphs[0]
            p.runs[0].text = t
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            return
    raise AssertionError("eyebrow")


def add_notes(slide, t):
    tf = slide.notes_slide.notes_text_frame
    tf.text = (tf.text + "\n\n" + t) if tf.text else t


def video(slide, name, x, y, w, caption):
    """Embedded, playable clip with a poster frame; caption under it."""
    from PIL import Image
    stem = name[:-4]
    poster = POST / f"{stem}.jpg"
    vw, vh = Image.open(poster).size
    h = w * vh / vw
    mv = slide.shapes.add_movie(str(MEDIA / name), Inches(x), Inches(y), Inches(w), Inches(h),
                                poster_frame_image=str(poster), mime_type="video/mp4")
    rect(slide, x, y, w, h, fill=None, line=MIDGRAY, lw=0.75)
    text(slide, x, y + h + 0.06, w, 0.5, caption, size=10)
    return y + h


def video_slot(slide, x, y, w, h, label):
    """Placeholder for a clip still to be supplied."""
    s = rect(slide, x, y, w, h, fill=PANEL, line=MAROON, lw=1.0)
    ln = s.line._get_or_add_ln()
    dash = etree.SubElement(ln, qn("a:prstDash"))
    dash.set("val", "dash")
    icon(slide, "video", x + 0.15, y + (h - 0.42) / 2, 0.42)
    text(slide, x + 0.7, y, w - 0.8, h, [[("VIDEO SLOT  ", True, MAROON), (label, False)]], size=10,
         anchor=MSO_ANCHOR.MIDDLE)


def tag(slide, x, y, label, fill=GOLD, color=INK, w=None):
    w = w or (0.09 * len(label) + 0.2)
    rect(slide, x, y, w, 0.22, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(slide, x, y, w, 0.22, label, size=8, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def new(title, eb):
    s = content_slide(title)
    eyebrow(s, eb)
    return s


# ------------------------------------------------------------------ Chinmay's revised slides (22 Nov)
s = S[12]  # where it fails
replace(s, "32 mm", "44 mm")
set_eyebrow(s, "RESULTS · WHERE IT FAILS · COURSE: L2/L9 MDP")
set_para(s, "median final sideways offset", "median final sideways offset of the base policy to the target port (n = 144)")
set_para(s, "Across 380 logged rollouts", "4–6 cm away in total, mostly sideways: median 44 mm lateral. The plug stops level "
         "with the port face, so the open problem is lateral alignment.")
set_para(s, "Base policy xvla-aic-v4", "Base xvla-aic-v4, n = 144 eval episodes, 4 batches pooled (95% CI of median "
         "40.4–47.7 mm); face height: 141 failed base episodes, median −2.2 mm above the entrance; reward_analysis/e3_offsets.py")
for sh in list(s.shapes):  # move the takeaway box up a little to free a video slot
    if Inches(3.2) < sh.top < Inches(3.5):
        sh.top = sh.top - Inches(0.25)
video_slot(s, L, 4.05, 9.26, 0.7, "a typical X-VLA sideways miss, to be picked from the evaluation recordings")
s.notes_slide.notes_text_frame.text = (
    "Chinmay, revised slide 11. Where X-VLA fails: median final sideways offset to the target port is 44 mm (n = 144, "
    "4 batches pooled, 95% CI of the median 40.4–47.7 mm). Neighbouring SFP ports are 23 mm apart; the scorer allows 5 mm "
    "sideways for a partial insertion. The plug ends 4–6 cm away in total, mostly sideways, and stops level with the port "
    "face (141 failed base episodes, median −2.2 mm above the entrance). This replaces the earlier 32 mm (n = 36). "
    "Reconciliation: the 4–6 cm in Ibraheem's failure analysis is the same distance; for X-VLA it is lateral, not depth.")

s = S[14]  # measurement
set_eyebrow(s, "METHODS · EVALUATION · COURSE: L8 REPORTING RL RESULTS")
text(s, L, 4.22, 9.26, 0.4, "Different protocol from the earlier single-run table: per-episode means, base re-run in "
     "every batch.", size=13)
add_notes(s, "Revised: the protocol differs from the single-run 36-trial table (one run per policy, total score).")

s = S[15]  # reward bug
replace(s, "His RL reward", "Original RL reward")
set_eyebrow(s, "METHODS · REWARD BUG AND FIX · COURSE: L12 REWARD DESIGN (GT SOPHY)")

s = S[16]  # seeded table
set_eyebrow(s, "RESULTS · PRE-REGISTERED · 12 BOARDS × 3 RUNS, BASE RE-RUN IN EACH BATCH · COURSE: L8")
replace(s, "Recap v1 retrained (his recipe)", "Recap v1 retrained (original recipe)")
replace(s, "DSRL ±1.5, his reward", "DSRL ±1.5, original reward")
replace(s, "over his", "over original")
for sh in list(s.shapes):  # push the four criteria cards down to make room for the re-run line
    if Inches(3.2) < sh.top < Inches(4.7):
        sh.top = sh.top + Inches(0.4)
        if sh.height > Inches(1.3):
            sh.height = Inches(1.05)
        elif sh.height > Inches(0.7):
            sh.height = Inches(0.45)
    elif Inches(4.7) <= sh.top < Inches(4.9):
        sh.top = Inches(5.08)
text(s, L, 3.15, 9.26, 0.45, [[("Single-run results from the previous slide, re-run in the same batch: ", False),
                               ("Recap r1 checkpoint +1.0 [−6.3, 8.4]; Nov 15 DSRL agent +1.9 [−4.3, 8.0].", True),
                               (" Highest single runs, not significant.", False)]], size=11)
add_notes(s, "Revised: Chinmay re-ran Ibraheem's own Recap round 1 checkpoint (xvla-recap-v1) and the Nov 15 DSRL agent "
             "in the same batch as the base: +1.0 [−6.3, 8.4] and +1.9 [−4.3, 8.0]. The single-run lead of Recap r1 "
             "(1163 vs 1052) does not hold up under the paired comparison.")

s = S[19]  # lessons
set_eyebrow(s, "DISCUSSION · COURSE: L8 · L12 · L15")
set_para(s, "+8.5 became", "Our best DSRL agent: +8.5 → +1.9 on re-run.")
set_para(s, "Scripted demos are", "Scripted demos are likely the bottleneck.")
set_para(s, "github.com/Chinmay", "github.com/Chinmay-Prashanth/eee598-aic-dsrl · ibraheem-111/eee598-aic")

s = S[17]  # base vs DSRL still
set_eyebrow(s, "RESULTS · EVAL BOARD 6 (SFP), ONE RECORDING EACH, REAL TIME · COURSE: L8 (ONE RUN ≠ EVALUATION)")
tag(s, L + 0.1, 1.22, "VIDEO SLOT · Chinmay's board-6 clips; still shown", fill=GOLD, w=3.2)
add_notes(s, "Video slot: replace the still with Chinmay's two board-6 recordings (runs/videos/board6*) when he sends them.")

s = S[18]
set_eyebrow(s, "RESULTS · WHY DSRL DIDN'T HELP · COURSE: L10 DEADLY TRIAD · L15 SAC")
s = S[13]
set_eyebrow(s, "METHODS · THE RL PROBLEM · COURSE: L2/L3 MDP · L15 · L16 · L13")

# ------------------------------------------------------------------ Ibraheem's slides: reconciliations
s = S[2]
set_eyebrow(s, "INTRODUCTION · TASK AND PROTOCOL")
s = S[3]
set_eyebrow(s, "METHODS · DEMONSTRATION DATA")
s = S[4]
set_eyebrow(s, "RESULTS · IMITATION LEARNING")
s = S[5]
set_eyebrow(s, "METHODS · PORT LOCALIZATION")
s = S[6]
set_title(s, "Three RL methods on top of X-VLA")
set_eyebrow(s, "METHODS · REINFORCEMENT-LEARNING SETUP")
replace(s, "X-VLA itself is fine-tuned, conditioned on “Advantage: positive”.",
        "All 0.9B parameters of X-VLA are fine-tuned, conditioned on “Advantage: positive”.")
replace(s, "A bounded twist (±2 cm/s) is added to X-VLA's chunk.", "X-VLA frozen; a bounded twist (±2 cm/s) is added to its chunk.")
replace(s, "SAC picks X-VLA's flow-matching start noise (30×20, ±1.5σ).", "X-VLA frozen; SAC picks its flow-matching start noise (30×20, ±1.5σ).")
for sh in s.shapes:  # un-highlight the Recap card (no longer presented as the winner)
    if sh.shape_type == 1 and sh.fill.type == 1 and sh.fill.fore_color.rgb == LIGHTGOLD:
        sh.fill.fore_color.rgb = PANEL
add_notes(s, "Revised title (was: Three RL methods on a frozen X-VLA). Only residual SAC and DSRL keep X-VLA frozen; "
             "Recap fine-tunes all of it (train_xvla.sh: vision and language encoders unfrozen).")

s = S[7]
set_title(s, "Single-run scores, 36 trials each")
set_eyebrow(s, "RESULTS · ONE RUN PER POLICY, NO CONFIDENCE INTERVAL")
chart = next(sh for sh in s.shapes if sh.has_chart).chart
pt = chart.plots[0].series[0].points[6]
pt.format.fill.solid()
pt.format.fill.fore_color.rgb = GRAY
set_para(s, "Total AIC score, max 3600.", "Total AIC score, max 3600; one run each. All Recap and RL rows start from the "
         "same fine-tuned X-VLA. Seeded re-run of the top two: next slide.")
add_notes(s, "Revised title (was: Recap round 1 is the best policy; online RL stays within noise). Chinmay's paired "
             "re-run of the Recap r1 checkpoint gives +1.0 [−6.3, 8.4] per episode over the base, and of the DSRL agent "
             "+1.9 [−4.3, 8.0]: neither is significant.")

s = S[8]
set_eyebrow(s, "RESULTS · RECAP ITERATIONS")
s = S[9]
set_eyebrow(s, "RESULTS · SUCCESS BY PORT AND MOUNT")
set_para(s, "Most failures end 4–6 cm", "Most failures end 4–6 cm from the port; for X-VLA, mostly sideways (previous slide).")
add_notes(s, "Reconciliation: the original note says misses are depth and the last centimetres; that came from SmolVLA v3. "
             "For X-VLA, Chinmay's offsets (n = 144) show the miss is mostly lateral (median 44 mm), with the plug level "
             "with the port face.")

s = S[11]
set_eyebrow(s, "ARTIFACTS")
replace(s, "Best model", "Highest single-run score")
add_notes(s, "Revised: xvla-recap-v1 is labelled by its single-run score, not as the best model; the seeded re-run "
             "shows no significant gain over the base. The Hugging Face card should be updated to match.")

for i, e in ((20, "INTRODUCTION · THE CASCADE"), (21, "RESULTS · REAL ARMS, SEPARATE SYSTEM · WHY THE LAYERS ARE SEPARATE"),
             (22, "DISCUSSION · WHAT LEARNS FROM WHAT"), (23, "METHODS · PROPOSED, NOT BUILT · RL AT THE LOW LEVEL")):
    set_eyebrow(S[i], e)
add_notes(S[20], "In the unified deck, the next slides map this cascade onto AIC: X-VLA plans 1 s twist chunks at 10 Hz on "
                 "top of AIC's impedance controller (500 Hz controller manager) with a fixed stiffness (see the control "
                 "interface slide).")

# ------------------------------------------------------------------ new slides
N = {}
s = N["hard"] = new("Why the last few centimetres are hard", "INTRODUCTION · THE PROBLEM")
for i, (n, lab, c) in enumerate([("5 mm", "sideways tolerance the scorer allows for a partial insertion", MAROON),
                                  ("23 mm", "spacing between neighbouring SFP ports: a miss this size lands in the wrong one", INK),
                                  ("1039/1200", "CheatCode with ground-truth poses: the task is solvable when the port is known", INK)]):
    bignum(s, L + i * 3.13, 1.25, 2.9, n, lab, color=c, num_size=38, label_size=12)
rect(s, L, 3.3, 9.26, 0.85, fill=PANEL)
icon(s, "crosshairs", L + 0.15, 3.42, 0.6)
text(s, L + 0.95, 3.3, 8.2, 0.85, "From cameras alone, the policy must place the plug within millimetres, in a port that "
     "looks like its neighbours, inside 60 s and without exceeding 30 N.", size=13, anchor=MSO_ANCHOR.MIDDLE)
source(s, "Tolerance and spacing: Chinmay's analysis (scorer terms, board geometry). CheatCode: 1039/1200 on the seed-100 "
          "set. Limits: AIC trial settings.")
notes(s, "Why the task is hard. The scorer gives partial credit only within 5 mm sideways, and neighbouring SFP ports are "
         "23 mm apart, so a policy that is a few centimetres off lands in the wrong port. CheatCode, which reads ground-truth "
         "poses, scores 1039/1200 on the seed-100 set, so the difficulty is perception and precise alignment, not the arm.")

s = N["questions"] = new("Three questions", "INTRODUCTION · QUESTIONS AND CONTRIBUTIONS")
for i, (ic, k, t, b) in enumerate([
        ("graduationcap", "IMITATION", "How far does imitation get?", "SmolVLA, ACT and a fully fine-tuned X-VLA on 585 "
                                                                       "scripted demonstrations."),
        ("dice", "RL ON THE PLAN", "Does RL on top of the VLA help?", "Residual SAC, DSRL and Recap, evaluated with "
                                                                       "seeds and paired confidence intervals."),
        ("sliders", "RL AT THE LOW LEVEL", "Where else could RL act?", "A proposed Q-network that sets speed, stiffness "
                                                                       "and force under the VLA (design only).")]):
    x = L + i * 3.13
    rect(s, x, 1.2, 2.95, 2.6, fill=PANEL)
    icon(s, ic, x + 0.18, 1.35, 0.5)
    text(s, x + 0.18, 1.95, 2.6, 0.25, k, size=9, bold=True, color=MAROON)
    text(s, x + 0.18, 2.2, 2.6, 0.7, t, size=14, bold=True)
    text(s, x + 0.18, 2.95, 2.6, 0.8, b, size=11)
text(s, L, 4.05, 9.26, 0.6, "Team: Muhammad Ibraheem (imitation, port localization, RL runs), Chinmay Prashanth "
     "(DSRL, reward analysis, seeded evaluation), control-stack design (last section).", size=11, color=GRAY)
notes(s, "The three questions the deck answers, one per section of results. Attribution line: check with the team; "
         "the control slides' author was not stated in the source file.")

s = N["interface"] = new("Our control stack in AIC", "METHODS · SIMULATOR AND CONTROL INTERFACE")
layers = [("brain", "X-VLA policy", "3 wrist cameras + TCP pose, velocity, wrist F/T → a 1 s chunk of 6-D TCP twists at "
                                    "10 Hz, replanned every 1 s", "LEARNED"),
          ("gears", "AIC impedance controller", "Cartesian impedance, controller manager at 500 Hz; our policy sends a "
                                                "fixed stiffness: 100 N/m translation, 50 rotation", "FIXED GAINS"),
          ("robot", "UR5e in Gazebo", "official AIC stack, ROS 2 Kilted, real time; 60 s per trial; stops at 6 cm/s, "
                                      "0.4 rad/s, 30 N, 40 cm drift", "SIMULATED")]
for i, (ic, t, b, tg) in enumerate(layers):
    y = 1.2 + i * 1.12
    rect(s, L, y, 6.2, 0.95, fill=LIGHTGOLD if i == 1 else PANEL)
    icon(s, ic, L + 0.15, y + 0.2, 0.55)
    text(s, L + 0.85, y + 0.1, 3.5, 0.3, t, size=13, bold=True)
    tag(s, L + 4.75, y + 0.12, tg, w=1.3, fill=GOLD if i == 0 else MIDGRAY)
    text(s, L + 0.85, y + 0.42, 5.2, 0.5, b, size=10)
    if i < 2:
        arrow(s, L + 3.1, y + 0.95, L + 3.1, y + 1.12, color=MAROON, w=1.25)
card(s, 6.85, 1.2, 2.78, 3.19, "Where RL can act", ["Residual SAC: adds to the twist.", "DSRL: picks X-VLA's start "
     "noise.", "Recap: retrains X-VLA.", [("Not yet tried: ", True), ("the stiffness field of each command "
     "(MotionUpdate.target_stiffness), as in the low-level design.", False)]], ic="sliders", body_size=10)
text(s, L, 4.6, 9.26, 0.35, "X-VLA's reaction latency in AIC has not been measured yet (run E6).", size=10, color=GRAY)
notes(s, "This is the AIC column of the cascade: a learned planner at 10 Hz over fixed-gain impedance control. Sources: "
         "external/aic/aic_bringup/config/aic_ros2_controllers.yaml (controller manager 500 Hz, impedance mode); "
         "policy/aic_smolvla/RunSmolVLA.py line 100 (target_stiffness diag 100,100,100,50,50,50). Latency to be measured.")

s = N["policies"] = new("Three imitation policies", "METHODS · POLICIES")
for i, (ic, t, b) in enumerate([
        ("layergroup", "SmolVLA 450M", "Vision-language-action model; only its 100M action expert trains (vision frozen). "
                                       "Datasets v1–v3, 15k–40k steps."),
        ("cubes", "ACT 52M", "Action-chunking transformer trained from scratch on v3, 50k steps."),
        ("microchip", "X-VLA 0.9B", "All 879M parameters fine-tuned, Florence-2 vision encoder included; v4, 60k steps. "
                                    "Base for every RL method.")]):
    x = L + i * 3.13
    card(s, x, 1.2, 2.95, 2.4, t, b, ic=ic, body_size=11, fill=LIGHTGOLD if i == 2 else PANEL)
text(s, L, 3.85, 9.26, 0.6, "All three predict chunks of TCP twists from the three wrist cameras and the robot state; "
     "scores are compared on the same 36-trial set (results section).", size=12)
notes(s, "Moved from the imitation-learning slide: model sizes and what trains. SmolVLA trains only its 100M action expert "
         "(vision frozen); X-VLA trains all 879M parameters, Florence-2 included.")

s = N["dsrl"] = new("DSRL: steer a frozen policy", "METHODS · KEY PAPER · COURSE: TOPIC 11 · WAGENMAKER ET AL., CORL 2025")
boxes = [("STATE Sₜ", "wrist cameras, robot state", PANEL, INK),
         ("RL: SAC ACTOR", "π_θ(w | Sₜ) picks the start noise", LIGHTGOLD, INK),
         ("ACTION", "noise w ∈ [−b, b]⁶⁰⁰", PANEL, INK),
         ("FROZEN VLA (X-VLA)", "flow-matching head turns w into a 1 s action chunk", MAROON, WHITE),
         ("ENVIRONMENT", "Gazebo runs the chunk", PANEL, INK)]
bw, gap = 1.62, 0.29
for i, (t, b, f, c) in enumerate(boxes):
    x = L + i * (bw + gap)
    rect(s, x, 1.25, bw, 1.25, fill=f, line=MAROON if i == 1 else None, lw=1.5)
    text(s, x + 0.1, 1.33, bw - 0.2, 0.3, t, size=9, bold=True, color=GOLD if f == MAROON else MAROON)
    text(s, x + 0.1, 1.65, bw - 0.2, 0.8, b, size=11, color=c)
    if i < 4:
        arrow(s, x + bw + 0.03, 1.87, x + bw + gap - 0.03, 1.87)
c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(L), Inches(2.75), Inches(R), Inches(2.75))
nostyle(c)
c.line.color.rgb = MAROON
c.line.width = Pt(1.25)
etree.SubElement(c.line._get_or_add_ln(), qn("a:prstDash")).set("val", "dash")
text(s, L, 2.9, 9.26, 0.35, "← reward Rₜ₊₁, Sₜ₊₁ → replay buffer → critics Q_φ(Sₜ, w) → update π_θ (off-policy)",
     size=12, color=MAROON)
text(s, L, 3.5, 9.26, 0.6, "The VLA's weights never change. RL only chooses where its sampler starts, so actions stay "
     "close to the demonstrations.", size=13)
source(s, "Wagenmaker et al., “Steering Your Diffusion Policy with Latent Space Reinforcement Learning”, CoRL 2025; "
          "code: dsrl_pi0 (jaxrl2)")
notes(s, "Chinmay, revised slide 10 (original title: DSRL: steer a frozen policy through its noise). DSRL keeps the VLA "
         "frozen and learns a SAC actor over the flow-matching start noise; transitions go to a replay buffer and twin "
         "critics, updated off-policy.")

s = N["vid_imit"] = new("Imitation in the simulator", "RESULTS · IMITATION LEARNING · RECORDINGS")
video(s, "ft_v3_trial4_partial_insertion.mp4", L, 1.25, 4.45, "SmolVLA v3, held-out board (seed 100, trial 4): "
      "partial insertion.")
video(s, "xvla_full_insertion_s101_trial8.mp4", 5.18, 1.25, 4.45, "X-VLA, held-out board (seed 101, trial 8): full "
      "insertion.")
source(s, "Click to play. Different trials; recordings from the 36-trial evaluations (docs/media).")
notes(s, "Two embedded clips: SmolVLA v3's partial insertion and X-VLA's full insertion, each from the held-out "
         "evaluation sets. They are different trials, chosen to show each policy's best behaviour.")

s = N["vid_recap"] = new("Same trial: X-VLA vs Recap r1", "RESULTS · RECAP · RECORDINGS")
video(s, "xvla_partial_insertion_s100_trial11.mp4", L, 1.25, 4.45, "X-VLA (base), seed 100, trial 11: partial insertion.")
video(s, "recap_full_insertion_s100_trial11.mp4", 5.18, 1.25, 4.45, "X-VLA + Recap round 1, same trial: full insertion "
      "at 51 s.")
source(s, "Click to play. One trial each, not an evaluation: the seeded re-run shows no significant gain (+1.0 [−6.3, 8.4]).")
notes(s, "Same board and target (seed 100, trial 11): the base X-VLA gets a partial insertion; Recap round 1 a full "
         "insertion. A single trial; Chinmay's paired re-run of the Recap r1 checkpoint shows no significant gain.")

s = N["limits"] = new("Limitations and next steps", "DISCUSSION · LIMITATIONS AND NEXT STEPS")
text(s, L, 1.15, 4.4, 0.3, "Limitations", size=14, bold=True, color=MAROON)
for i, (ic, t) in enumerate([("chartcolumn", "0–3 insertions per 36 episodes: methods are separable only with seeds."),
                             ("clock", "Real-time Gazebo limits RL to ~300 episodes per run."),
                             ("triangleexclamation", "The original RL reward paid +50 for wrong-port insertions."),
                             ("brain", "DSRL's SAC critic diverges (Q 10⁶–10⁷)."),
                             ("robot", "Simulation only; the low-level RL design is not built.")]):
    y = 1.55 + i * 0.66
    icon(s, ic, L, y, 0.42, circle=GRAY)
    text(s, L + 0.55, y, 3.85, 0.62, t, size=11)
text(s, 5.2, 1.15, 4.4, 0.3, "Next", size=14, bold=True, color=MAROON)
for i, (ic, t) in enumerate([("crosshairs", "Fix lateral alignment: port estimate with three-camera agreement."),
                             ("boxarchive", "More varied demonstrations; SC-targeted demos and evaluation."),
                             ("medal", "Recap round 3 with the fixed reward and demos as the majority."),
                             ("bolt", "TD3 (no entropy term) for DSRL; 10 seeds per method."),
                             ("sliders", "Test low-level RL in AIC via the stiffness field.")]):
    y = 1.55 + i * 0.66
    icon(s, ic, 5.2, y, 0.42)
    text(s, 5.75, y, 3.88, 0.62, t, size=11)
notes(s, "Merged from Ibraheem's limitations slide, Chinmay's lessons and future work, and the control slides' design. "
         "Original limitations: 36 trials per policy; insertion counts of 0–3 cannot separate methods; scores carry "
         "±30–40 noise per 12-trial set. Real-time Gazebo limits RL to ~300 episodes per run. Recap's value function sees "
         "privileged state only (correlation 0.20–0.55). Simulation only. Original next steps: Recap round 3 (demos as "
         "majority, top 10–15% positive, image-conditioned value function; evaluate round 2 at 20k), larger evaluation "
         "(~60 boards), SC-targeted demos, 4+ parallel simulators on a 24 GB GPU, port estimate with three-camera "
         "agreement. Chinmay's future work: TD3 (no entropy term), more varied demos, more seeds (L8: 10).")

# ------------------------------------------------------------------ order, sections, deletions
order = [("Title", [S[1]]),
         ("Introduction", [S[2], N["hard"], S[20], N["questions"]]),
         ("Methods", [N["interface"], S[3], N["policies"], S[5], S[6], N["dsrl"], S[13], S[15], S[14], S[23]]),
         ("Results", [S[4], N["vid_imit"], S[7], S[16], S[8], N["vid_recap"], S[18], S[17], S[12], S[9], S[21]]),
         ("Discussion", [S[22], S[19], N["limits"], S[11]]),
         ("Closing", [S[24]])]
lst = prs.slides._sldIdLst
by_part = {sid.get(qn("r:id")): sid for sid in lst}
slide_rid = {}
for sid in lst:
    slide_rid[prs.part.related_part(sid.get(qn("r:id")))] = sid
keep = [sl for _, group in order for sl in group]
for sid in list(lst):
    lst.remove(sid)
for sl in keep:
    lst.append(slide_rid[sl.part])
dropped = [sid for part, sid in slide_rid.items() if sid not in list(lst)]
for sid in dropped:  # the old limitations slide (merged)
    prs.part.drop_rel(sid.get(qn("r:id")))

# PowerPoint sections
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
pres = prs.part._element
ext_lst = pres.find(qn("p:extLst"))
if ext_lst is None:
    ext_lst = etree.SubElement(pres, qn("p:extLst"))
for ext in list(ext_lst):
    if ext.get("uri") == "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}":
        ext_lst.remove(ext)
ext = etree.Element(qn("p:ext"), uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
ext_lst.insert(0, ext)
sec_lst = etree.SubElement(ext, f"{{{P14}}}sectionLst", nsmap={"p14": P14})
import uuid
for name, group in order:
    sec = etree.SubElement(sec_lst, f"{{{P14}}}section", name=name, id="{" + str(uuid.uuid4()).upper() + "}")
    ids = etree.SubElement(sec, f"{{{P14}}}sldIdLst")
    for sl in group:
        etree.SubElement(ids, f"{{{P14}}}sldId", id=slide_rid[sl.part].get("id"))

prs.save(OUT_PATH)
print("saved", OUT_PATH, len(prs.slides), "slides")
