# Literature review: SmolVLA + DSRL for AIC cable insertion (2026-10-15)

Compiled with web research for the EEE 598 project. Paper numbers come from summaries of the papers'
pages, not a full read of each PDF; check a figure against the paper before citing it in the report.
Section (d) lists what could not be verified.

**Where we stand:** CheatCode demos (253 successful) → fine-tuned SmolVLA reaches the target port
(median 6 cm short) but never inserts; plain-language prompts + a target-port image made no difference.

## (a) Most actionable lessons, ranked

1. **The bottleneck is getting to the port, not RL.** A 6 cm median miss can't be fixed by insertion
   tuning. No top AIC team solved the task end to end with a learned policy: all top-10 teams used the
   Intrinsic Vision Model for "submillimeter, high precision pose detection" and 90% used classical
   force-guided insertion (spiral search, alignment corrections)
   ([Intrinsic results](https://www.intrinsic.ai/blog/posts/ai-for-industry-challenge)). The clean
   version of our project is hybrid: perception estimates the port pose, the learned/RL policy works
   in port-relative coordinates, and RL owns the last few centimetres.
2. **Relabel actions with commanded targets, not measured TCP motion.** Measured-motion labels teach
   "stop at contact" (our ~50% near-zero frames). DexForce: policies trained without force-informed
   targets have near-zero success vs 76% with them ([2501.10356](https://arxiv.org/abs/2501.10356)).
   Diffusion Policy: idle actions make BC get stuck, and position actions beat velocity actions
   ([2303.04137](https://arxiv.org/html/2303.04137v5)).
3. **Re-plan more often.** 50 steps at 10 Hz is 5 s open loop. Execute ~5–10 steps per inference, or
   use Real-Time Chunking / temporal ensembling. Diffusion Policy found an 8-step horizon best; AIC
   participants were advised 25–50-step chunks with 3–5 observation steps
   ([Discourse](https://discourse.openrobotics.org/t/my-first-results-pi0-5-vla-policy/53670)).
4. **Raise image resolution.** The one AIC participant with a documented learned full insertion (ACT,
   ~1000 episodes) went from partial to full insertion by doubling resolution from 288×256 — our exact
   resolution ([same thread](https://discourse.openrobotics.org/t/my-first-results-pi0-5-vla-policy/53670)).
5. **DSRL won't create insertion from a 0%-insertion base policy.** DSRL only reaches actions inside
   the base policy's support ([2506.15799](https://arxiv.org/html/2506.15799v2)), and noise-space RL is
   inefficient at low initial success with sparse rewards
   ([UniSteer, 2605.10821](https://arxiv.org/html/2605.10821v1)). Get BC insertion above zero first, or
   use residual RL (ResiP: peg-in-hole 5% → 99%, [2407.16677](https://arxiv.org/html/2407.16677v4)).
6. **Add force/torque, fused late.** ForceVLA: π0 37.3% → 60.5% average over five contact tasks; naive
   concatenation added ~3 points and early fusion into the VLM failed
   ([2505.22159](https://arxiv.org/html/2505.22159v3)). Tare it: the AIC wrist sensor reads ~20 N at
   rest from tool mass ([Rocky's thread](https://discourse.openrobotics.org/t/rockys-open-source-build-thread-ai-for-industry-challenge/53155)).
   *Verified in AIC's scorer:* the force penalty subtracts the tare offset
   (`aic_scoring/src/ScoringTier2.cc` lines 434–439), so the resting bias does not count against us.
7. **Add recovery data and randomize starts.** DART-style noise in the scripted expert, randomized grasp
   offsets, fewer long pauses. AIC participants report memorized trajectories and models ignoring vision.
8. **Our null result for language/reference images is expected.** VLAs largely ignore language
   ([LIBERO-Plus, 2510.13626](https://arxiv.org/html/2510.13626v1)); context can't fix a precision or
   labeling problem.

## (b) Findings

### AIC challenge write-ups and results
- **Official** ([Intrinsic](https://www.intrinsic.ai/blog/posts/ai-for-industry-challenge)): ~5,000
  developers, 115 countries. 1st Team Flex (USA), 2nd TCS-PLGRM (India); only those two scored 100% on
  the physical workcell. Top-10: Intrinsic Vision Model for pose, 90% classical force-feedback insertion,
  a hard split between perception and action; cable handling ("flinging" the cable clear) mattered.
  Phases: Gazebo qualification (~160 teams) → top 30 in Flowstate → top 10 on the real workcell
  ([Datameister](https://datameister.ai/blog/intrinsic-ai-for-industry-challenge-qualifying-first/)).
- **Datameister** (1st in qualifying, 293.38): chained step-specific policies on their production
  perception library, one metric per sub-problem.
- **b-robotized / Kim McGuire** (10th of 160, 250/300) ([blog](https://www.mcguirerobotics.com/blog/2026/09/01/going-through-the-ai-for-industry-challenge/)):
  OpenCV + solvePnP for the board, YOLO for ports, HIL-SERL for insertion; HIL-SERL's live-interaction
  dependence transferred poorly to sim-only evaluation.
- **fm-sw** (54th) ([page](https://fm-sw.com/project-intrinsic.html)): classical vision + F/T pipeline.
- **Discourse reports:**
  - jlamperez: π0.5 with ~100 episodes understood the scene but never inserted. ACT with ~1000 episodes
    (one model per configuration) fully inserted on trial 1 (75 points). Fixes that mattered: a joint-order
    bug that made models ignore vision, keeping the port in view, and doubling image resolution.
  - Pranav_Saroha: with 160 episodes only ACT partially succeeded; π0.5 and SmolVLA did not. A diffusion
    policy on a 6,600-episode DAgger dataset scored 9 vs ~60 for ACT and headed to the wrong port
    ([thread](https://discourse.openrobotics.org/t/6600-demos-recorded-uplaoded-on-hugging-face/55167),
    [data](https://huggingface.co/datasets/bha-51/aic-dagger-data)).
  - Qualification learnings ([thread](https://discourse.openrobotics.org/t/learnings-from-the-qualification-phase-and-what-next/54899)):
    SmolVLA on ~550 episodes peaked around 40, with oscillation; Isaac Lab RL failed for one team, whose
    classical stack got 80% SFP / 0% SC (~75 points); ACT on 300 classical trajectories plateaued.
  - rajeshramana: ACT for approach/alignment + scripted FSM for insertion and retries; works for SFP,
    poorly for SC ([thread](https://discourse.openrobotics.org/t/policy-question-what-have-you-tried-so-far/54365)).
  - Rocky Shao: ground-truth expert, force-aware FSM, ACT; found the ~20 N gravity bias and that
    compliance gains beat retreat logic ([repo](https://github.com/Rocky0Shao/IntrinsicAIChallenge)).
- **Pattern:** pure VLAs (π0.5, SmolVLA) did worst; ACT was the best learned baseline; SC was harder
  than SFP everywhere; every winning stack was hybrid.

### VLAs fine-tuned for precise insertion
- ForceVLA ([2505.22159](https://arxiv.org/html/2505.22159v3)): ~50 demos/task; vision-only π0 weak on
  plug/USB insertion; force fusion up to 80%.
- ReTac-ACT ([2603.09565](https://arxiv.org/abs/2603.09565)): 90% NIST peg-in-hole, 80% at 0.1 mm
  clearance, with tactile fusion; beats vision-only and generalist baselines.
- VTLA ([2505.09577](https://arxiv.org/abs/2505.09577)): vision-tactile VLA trained on simulated
  insertion data, >90% on unseen pegs.
- OpenVLA-OFT ([2502.19645](https://arxiv.org/abs/2502.19645)): continuous actions, chunking and
  parallel decoding were key (not insertion-specific).
- No paper found reports a vision-only VLA fine-tune reliably doing mm-clearance insertion.

### "Stops short" in behavior cloning
- Idle actions: Diffusion Policy §4.3 (policies get stuck unless idle actions are removed).
- Position vs velocity: Diffusion Policy §4.2 (position control consistently better).
- Commanded/force-informed targets: DexForce; Adaptive Compliance Policy (virtual target pose +
  stiffness, >50% improvement, [2410.09309](https://arxiv.org/pdf/2410.09309)); Wrench-ACT (position-ACT
  6% peg / 12% connector vs 74% / 80% with wrench actions, [2609.37552](https://arxiv.org/html/2609.37552v1)).
- Chunk length: Diffusion Policy (8 steps best); Real-Time Chunking for flow VLAs without retraining
  ([2506.07339](https://arxiv.org/abs/2506.07339)); ACT temporal ensembling ([2304.13705](https://arxiv.org/pdf/2304.13705)).
- Residual RL on BC: ResiP, peg-in-hole 5% → 99%; BC alone saturates at 56% even with 10k demos.
- Copycat / causal confusion ([Wen et al. 2020](https://proceedings.neurips.cc/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf)):
  proprioception dominates vision — the same failure reported in AIC.

### Force/torque in VLAs and BC
- ForceVLA: +23.2 points over π0; late concatenation 60% vs 80% with their MoE fusion; early fusion 0%.
- FoAR ([2411.15753](https://arxiv.org/abs/2411.15753)): contact predictor gates force features.
- Reactive Diffusion Policy ([2503.02881](https://arxiv.org/abs/2503.02881)): slow chunk policy + fast
  force/tactile loop.
- LeCo ([2609.31439](https://arxiv.org/html/2609.31439)): visual policy + admittance control (10 Hz
  policy / 100 Hz control), 40 demos/task, 94% on four real connector tasks, ~30% lower force peaks.
- Caveat: mostly small-N real-robot studies; none uses SmolVLA.

### RL fine-tuning of VLAs and diffusion policies
- DSRL ([2506.15799](https://arxiv.org/html/2506.15799v2)): SAC in noise space, ~500K trainable
  parameters, base policy frozen. Real-world 20% → 90% in ~50 episodes; π0 ~10–11k steps/task. Limited
  to the base policy's support.
- Successors (2026, abstracts only): UniSteer adds human corrections at low base success; Noise Space
  Learning tests SmolVLA and π0.5 ([2610.13943](https://arxiv.org/html/2610.13943)); DLSRL argues
  noise steering alone is limiting ([2609.11270](https://arxiv.org/abs/2609.11270)).
- Residual: ResiP (PPO, IsaacGym); ResFiT, ~200× more sample-efficient than PPO residual RL
  ([2509.19301](https://arxiv.org/html/2509.19301v2)); Policy Decorator ([2412.13630](https://arxiv.org/abs/2412.13630)).
- Full fine-tuning: PA-RL ([2412.06685](https://arxiv.org/abs/2412.06685)); ConRFT 96.3% in 45–90 min
  with human intervention ([2502.05450](https://arxiv.org/abs/2502.05450)); HIL-SERL USB insertion 100%
  in 2.5 h on a real robot with interventions ([2410.21845](https://arxiv.org/abs/2410.21845)); RL-100
  ([2510.14830](https://arxiv.org/html/2510.14830v1)); RLinf-VLA ([2510.06710](https://arxiv.org/abs/2510.06710)).
- **Fits one 12 GB GPU (judgement):** DSRL/NSL over a frozen SmolVLA (inference + small SAC), or a
  small residual policy with PPO on state (port pose, TCP, F/T) in Isaac Lab with parallel envs. Full
  VLA PPO, ConRFT, PA-RL: not realistic. The real limit is environment throughput: Gazebo is one env
  at ~real time, so 10k+ steps takes hours.

### Data collection
- DART ([1703.09327](https://arxiv.org/abs/1703.09327)): inject noise into the expert's control so it
  demonstrates recovery; cheap with CheatCode. MimicGen ([2310.17596](https://arxiv.org/pdf/2310.17596))
  injects action noise (σ = 0.05). CCIL ([2310.12972](https://arxiv.org/abs/2310.12972)) synthesizes
  corrective labels. IntervenGen ([PDF](https://autolab.berkeley.edu/assets/publications/media/IROS24_2512_FI.pdf)).
- AIC: ground-truth scripted demos beat keyboard teleop; purely classical data plateaued; volume
  without diversity (6,600 DAgger episodes) didn't beat a smaller ACT dataset.
- Literature successes use ~40–50 demos/task, but with force input or RL on top; BC saturates on
  precision tasks.

### Language and goal-image conditioning
- LIBERO-Plus: ten VLAs largely insensitive to language.
- Octo ([2405.12213](https://arxiv.org/pdf/2405.12213)): goal images ~25% better than language on
  WidowX — but those are images of the finished scene, not a reference image of the target.
- Conditioning matters for choosing among identical ports, not for millimetre precision.

## (c) Lesson → change → evidence

| Lesson | Change | Evidence |
|---|---|---|
| Measured-motion labels teach "stop at contact" | Relabel with CheatCode's commanded poses (recovered in commands.npz); predict delta pose | DexForce; ACP; Wrench-ACT; Diffusion Policy §4.2 |
| Idle frames cause stalling | Drop/downweight near-still frames outside contact | Diffusion Policy §4.3 |
| 5 s open-loop chunks | Execute 5–10 steps per inference; RTC or temporal ensembling | Diffusion Policy; RTC; ResiP |
| 288×256 too coarse | Re-record at 512+ or crop around the port | jlamperez (AIC); SmolVLA |
| No force input | Tared 6-axis F/T as a late-fused state token (wrench.npz recovered) | ForceVLA; FoAR |
| Port localization is the bottleneck | Port-pose front end; port-relative observations/actions | Intrinsic results; Datameister; McGuire |
| No recovery data | DART noise in CheatCode; randomize grasp and start | DART; MimicGen; CCIL |
| DSRL needs nonzero base success | Get BC insertion > 0 first; shaped reward; or residual RL in Isaac Lab | DSRL; UniSteer; ResiP |
| SC harder and underrepresented | Balance SC demos; consider separate SC/SFP models | AIC Discourse |
| Language/reference image don't help | Stop spending effort on context | LIBERO-Plus |

## (d) Could not verify
- Team Flex's method (a "lateral swinging" SC insertion appeared only in a search snippet). No
  technical write-up from Team Flex or TCS-PLGRM was found.
- Paper figures come from page summaries, not full PDFs — check before citing, especially ForceVLA's
  ablation, Wrench-ACT's table, ResiP 5% → 99%, NSL and UniSteer figures.
- 2026 arXiv papers (Wrench-ACT, LeCo, DLSRL, NSL, UniSteer, ReTac-ACT) were read at abstract level only.
- ACT battery-insertion and temporal-ensembling figures were not verified.
- Discourse threads were partly visible; other approaches may be missing.
- Single-GPU feasibility (RL section) is a judgement, not reported by any paper.
