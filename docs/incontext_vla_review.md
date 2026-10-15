# In-context information for VLAs: verified literature review (2026-10-15)

Question: who has improved vision-language-action models (or close relatives) by giving them extra
information in their input at inference time (demos, retrieval, visual prompts, video, reasoning,
memory), and how?

Every paper below was checked by opening its arXiv abstract page (or official venue page) and
confirming title, authors, date and the attributed claim; numbers not in the abstract came from the
arXiv HTML full text or project page. Pages were read through a summarising fetch tool, so spot-check
any number before citing it. PDF links follow arXiv's `arxiv.org/pdf/<id>` pattern.

## Short answer
- **Demos in the prompt work only when the model is trained to use them.** RICL retrained π0-FAST on
  400 demos with 4 retrieved demos in context: new tasks 2.5% → 31.25% with no weight updates, and
  31.67% → 61.67% with fine-tuning. ICRT (1–3 demos) reached 79.2% on unseen tasks vs Octo 9.2% and
  OpenVLA 7.5%. Instant Policy (2 demos) reached 88.75% over 16 real tasks vs 34–41% for baselines.
- **Spatial prompts give the biggest gains for small data.** RT-Trajectory: 67% on unseen tasks vs 26%
  for goal images and 16.7% for language. TraceVLA: +10% over OpenVLA in SimplerEnv, 3.5× on real
  WidowX. HAMSTER: +20 points over OpenVLA. They tell the policy *where and how* to move.
- **Goal images and sketches help with ambiguity and distractors, not precision.** RT-Sketch matched
  baselines on plain tasks and won only with distractors or vague instructions. Interleave-VLA's
  image-in-instruction doubled generalisation to unseen objects, by identifying objects.
- **Reasoning in context helps when trained in.** ECoT +28 points over OpenVLA; CoT-VLA +17% real,
  +6% sim; π0.5's ablation shows training on subtask prediction matters even if not used at inference.
- **Live language corrections help long tasks by ~15–35 points** but need a human or a fine-tuned
  high-level policy (YAY Robot 15% → 50%; OLAF +20%).
- **Memory helps only history-dependent tasks.** HAMLET +47.2% on such tasks but 95.6 → 97.7% on LIBERO;
  MemoryVLA +26 on long real tasks. Naive frame stacking costs ~35% time and 3.6× memory (HAMLET), and
  frozen VLAs mostly ignore history ("Present but Not Remembered").
- **Counter-evidence: VLAs often ignore their context.** LIBERO-Plus: a blank instruction barely changes
  OpenVLA-OFT; swapping the target object drops success ~81% → ~0%. LIBERO-PRO: >90% → 0.0% under
  variation, outputs unchanged with corrupted instructions. A NeurIPS 2026 workshop study: random demo
  retrieval is competitive and retrieval-quality metrics don't predict success.
- **Common thread:** context helps when it carries decision-relevant information that is *not already in
  the observation* (where to go, which phase, contact state) and the model was trained with it.

## Papers (all verified)

| Paper | Year / venue | Context | How it's used | Key result | Retrain? | Code |
|---|---|---|---|---|---|---|
| [ICRT](https://arxiv.org/abs/2408.15980) (Fu et al.) | 2024 | 1–3 teleop demos as prompt | Causal transformer predicts next sensorimotor tokens | 79.2% unseen vs Octo 9.2%, OpenVLA 7.5% | Yes | [icrt.dev](https://icrt.dev/) |
| [RICL](https://arxiv.org/abs/2508.02062) (Sridhar et al.) | 2025, CoRL | 4 retrieved demos | Retrains π0-FAST on 400 demos with retrieved context | New tasks 2.5% → 31.25%; with fine-tune 31.67% → 61.67% | Yes (2×A100) | [code](https://github.com/ricl-vla/ricl_openpi) |
| [Instant Policy](https://arxiv.org/abs/2411.12633) (Vosylius & Johns) | ICLR 2025 | 1–2 demos as a graph | Graph diffusion trained on simulated pseudo-demos | 88.75% over 16 real tasks vs 34–41% | Yes (1 GPU, ~5 days) | [code](https://github.com/vv19/instant_policy) |
| [KAT](https://arxiv.org/abs/2403.19578) (Di Palo & Johns) | RSS 2024 | Keypoint/action token demos in text | GPT-4 Turbo as the policy, no training | 10 demos: 68% vs Diffusion Policy 10% | No | [Colab](https://www.robot-learning.uk/keypoint-action-tokens) |
| [ICI-VLA](https://arxiv.org/abs/2609.07581) (Yang et al.) | 2026 | Retrieved subtask demo segments | DTW retrieval + target-action masking | LIBERO 97.7%; RoboTwin 2.0 60.4% (+19.3); real 83.2% | Yes | — |
| [SynthICL](https://arxiv.org/abs/2606.08154) (Qian et al.) | 2026 | 1 RGB demo | Flow-matching transformer on synthetic data | 79% on 16 unseen real tasks | Yes | [project](https://synth-icl.github.io) |
| [SAIL](https://arxiv.org/abs/2603.08269) (Sato et al.) | IROS 2026 | Retrieved demos + feedback | VLM proposals + test-time tree search | Up to 95%; improves with compute | No | — |
| [ICLR (visual reasoning)](https://arxiv.org/abs/2603.07530) (Nguyen et al.) | IROS 2026 | Demos + predicted traces | Joint trace and action prediction | Qualitative in abstract | Yes | [project](https://toannguyen1904.github.io/ICLR) |
| [StellaVLA](https://arxiv.org/abs/2608.11671) (Xu et al.) | 2026 | Structured rationale demos | Dual action/language training | LIBERO 98.8%; LIBERO-Plus 85.1% | Yes | — |
| [REGENT](https://arxiv.org/abs/2412.04759) (Sridhar et al.) | ICLR 2025 | Retrieved neighbour states/actions | Retrieval-conditioned transformer | Matches generalists with 3× fewer params, 10× less data | Yes | [project](https://kaustubhsridhar.github.io/regent-research) |
| [RA-VLA](https://arxiv.org/abs/2608.25585) (Jang et al.) | ICML 2026 | Behaviour-aligned retrieval | Training-free adaptation | Qualitative in abstract | No | — |
| [When Does Retrieval Help?](https://neurips.cc/virtual/2026/161951) (Liu et al.) | NeurIPS 2026 workshop | Retrieved demos | Study of 4 retrieval methods | Random retrieval competitive; metrics don't predict success | — | [OpenReview](https://openreview.net/forum?id=iKYQg7eUJb) |
| [VIMA](https://arxiv.org/abs/2210.03094) (Jiang et al.) | ICML 2023 | Interleaved text + image prompts | Transformer over multimodal prompts | Up to 2.9× zero-shot | Yes | [project](https://vimalabs.github.io/) |
| [RT-Sketch](https://arxiv.org/abs/2403.02709) (Sundaresan et al.) | 2024 | Hand-drawn goal sketch | Trained on synthetic sketches | ≈ baselines on plain tasks; better with distractors/ambiguity | Yes | [project](http://rt-sketch.github.io) |
| [RT-Trajectory](https://arxiv.org/abs/2311.01977) (Gu et al.) | 2023 | Trajectory sketch on image | Hindsight trajectories drawn on training images | Unseen 67% vs goal image 26%, RT-1 16.7% | Yes | [project](https://rt-trajectory.github.io/) |
| [MOKA](https://arxiv.org/abs/2403.03174) (Liu et al.) | 2024 | Marks on images | VLM picks keypoints/waypoints | Qualitative in abstract | No | — |
| [Octo](https://arxiv.org/abs/2405.12213) (Ghosh et al.) | 2024 | Goal image or language | Transformer on 800K OXE trajectories | Fine-tunes in hours on consumer GPUs | Fine-tune | [project](https://octo-models.github.io) |
| [HAMSTER](https://arxiv.org/abs/2502.05485) (Li et al.) | 2025 | VLM-predicted 2D path | VLM draws path; 3D policy follows | +20 pts over OpenVLA | Yes | [project](https://hamster-robot.github.io/) |
| [TraceVLA](https://arxiv.org/abs/2412.10345) (Zheng et al.) | 2024/25 | Past motion trace on image | Fine-tuned VLA on 150K trajectories | +10% SimplerEnv; 3.5× real WidowX | Yes | — |
| [Interleave-VLA](https://arxiv.org/abs/2505.02152) (Fan et al.) | 2025 | Images interleaved in instructions | Auto-converted OXE instructions | 2× generalisation to unseen objects | Yes | [project](https://interleave-vla.github.io/Interleave-VLA-Anonymous/) |
| [Vid2Robot](https://arxiv.org/abs/2403.12943) (Jain et al.) | RSS 2024 | Human/robot prompt video | Video-to-state cross-attention | >20% over BC-Z | Yes | [project](https://vid2robot.github.io) |
| [Gen2Act](https://arxiv.org/abs/2409.16283) (Bharadhwaj et al.) | 2024 | Generated human video | Policy conditioned on generated video | 67% vs Vid2Robot 38% | Yes | [project](https://homangab.github.io/gen2act/) |
| [ECoT](https://arxiv.org/abs/2407.08693) (Zawalski et al.) | 2024 | Self-generated reasoning chain | OpenVLA trained on synthetic reasoning | +28 pts over OpenVLA | Yes | [project](https://embodied-cot.github.io) |
| [CoT-VLA](https://arxiv.org/abs/2503.22020) (Zhao et al.) | CVPR 2025 | Predicted subgoal image | Generate goal frame, then actions | +17% real, +6% sim | Yes | [project](https://cot-vla.github.io/) |
| [π0.5](https://arxiv.org/abs/2504.16054) (Black et al.) | 2025 | Predicted subtask text | Co-trained; subtask then action | Ablation: subtask training matters most | Yes | — |
| [YAY Robot](https://arxiv.org/abs/2403.12910) (Shi et al.) | 2024 | Live language corrections | Hierarchical; corrections fine-tune high level | Bag packing 15% → 50% | High-level | [project](https://yay-robot.github.io/) |
| [OLAF](https://arxiv.org/abs/2310.17555) (Liu et al.) | 2023 | Spoken corrections | LLM relabels actions | +20% average | Yes | [project](https://ut-austin-rpl.github.io/olaf/) |
| [RT-H](https://arxiv.org/abs/2403.01823) (Belkhale et al.) | 2024 | Language motions | Predict motion phrase, then action | Qualitative in abstract | Yes | [project](https://rt-hierarchy.github.io) |
| [LIBERO-PRO](https://arxiv.org/abs/2510.03827) (Zhou et al.) | 2025 | Counter-evidence | Perturbation benchmark | >90% → 0.0% under variation | — | [code](https://github.com/Zxy-MLlab/LIBERO-PRO) |
| [LIBERO-Plus](https://arxiv.org/abs/2510.13626) (Fei et al.) | CVPR 2026 | Counter-evidence | 10-model perturbation benchmark | Blank instruction ≈ no change; swapped target ~81% → ~0% | — | — |
| [MemoryVLA](https://arxiv.org/abs/2508.19236) (Shi et al.) | ICLR 2026 | Memory bank | Memory-conditioned diffusion expert | +26 on long real tasks | Yes | [project](https://shihao1895.github.io/MemoryVLA) |
| [SAM2Act](https://arxiv.org/abs/2501.18564) (Fang et al.) | 2025 | Spatial memory | Memory attention | MemoryBench 94.3% | Yes | [project](https://sam2act.github.io) |
| [HAMLET](https://arxiv.org/abs/2510.00695) (Koo et al.) | ICLR 2026 | Compressed history tokens | Memory adapter on GR00T N1.5 | +47.2% history tasks; LIBERO 95.6 → 97.7 | Adapter | [project](https://myungkyukoo.github.io/hamlet/) |
| [Present but Not Remembered](https://arxiv.org/abs/2607.03372) (Liao & Cao) | 2026 | Counter-evidence: history | Audit of frozen VLAs | History mostly redundant | — | — |
| [ForceVLA](https://arxiv.org/abs/2505.22159) (Yu et al.) | NeurIPS 2025 | 6-axis force/torque | Force-aware MoE fusion in π0 | +23.2% avg; up to 80% plug insertion | Yes | [site](https://sites.google.com/view/forcevla2025) |

## Lessons for our SmolVLA insertion project
Why our plain-text + port-image result was null: the camera already shows the port, so the image adds
little new information; constant text on a narrow single-task dataset is learned around (LIBERO-Plus,
LIBERO-PRO); and the hard part (the last 6 cm, contact) is not what these context types address.

Realistic on one 12 GB GPU, in order:
1. **Force/torque (and short proprioception history) in context** — ForceVLA +23.2%, up to 80% plug
   insertion. We have `wrench.npz` for all 306 demos; project a short force window into the state token
   and re-fine-tune. Cheap.
2. **Spatial prompts** (RT-Trajectory / TraceVLA / HAMSTER): overlay the projected port position or
   approach path on the camera images. Ground truth is free in sim (`tf.json`) for training; replace it
   with a keypoint detector at evaluation. Fits 12 GB (pixel-level change).
3. **Phase/subtask tokens** (π0.5, RT-H, ECoT-lite): label approach / align / insert / push from CheatCode's
   phases; add to text or as an auxiliary prediction head. No new data.
4. **Compressed history** (HAMLET-style) only if failures look like phase confusion.
5. **Few-shot / retrieval** (RICL, ICRT, Instant Policy, KAT): low priority. They target new tasks; with one
   task and 250 demos, fine-tuning beats in-context use (RICL's own numbers), and extra demo images are
   tight on 12 GB.

## Verification notes
- Dropped: "MoMa" (could not identify the paper); VLANeXt (2602.18532; page didn't match the claim).
- Abstracts without numbers (qualitative only): RA-VLA, ICLR (IROS 2026), MOKA, RT-H, π0.5.
- Venues from memory, not shown on abstract pages: HAMSTER, ECoT, SAM2Act, Octo.
- Seen in search but not opened, so not cited: TemporalFlow-VLA, μVLA, IntentVLA, CronusVLA, Afford-VLA.
