# Which foundation model, and which RL method? (2026-10-14)

Question: is SmolVLA + DSRL the right choice for the EEE 598 (RL for Robotics) project on the AIC
cable-insertion task, given a 12 GB RTX 5070 (24 GB borrowable but not preferred)?

**Short answer**
- **Model:** SmolVLA works but has the weakest prior. On 12 GB, **X-VLA-0.9B** is the best upgrade
  (robot-pretrained on 290K episodes, flow-matching head, full fine-tune fits); **Evo-1 (0.77B)** is
  the runner-up. Keep SmolVLA as the baseline we already have running. π0.5 fits on 12 GB only for
  LoRA at batch 1 and leaves no room for Gazebo during RL; it becomes the strongest option on 24 GB.
- **RL:** DSRL alone is too limited for precise insertion. Use **online off-policy actor-critic on a
  frozen VLA that both steers its noise (DSRL) and adds small force-aware residual corrections**,
  trained directly in the Gazebo evaluator with a dense reward from ground-truth poses. Compare
  DSRL-only vs residual-only vs both as the core course experiment.
- **Imitation learning is not RL.** Keep it to a small, auto-generated CheatCode demo set used as a
  prior and as replay data; the learning that matters for the course is online RL.

## 1. Measured fine-tuning memory on the RTX 5070

`lerobot-train`, 12 steps on `lerobot/svla_so101_pickplace` (2 cameras; AIC has 3, so add a little),
peak GPU memory of the training process incl. CUDA context. Card has 11.5 GiB usable.
Script: `scripts/bench_vla_train_memory.sh`.

| Model (params) | Mode | Batch | Peak | s/step | Fits 12 GB? |
|---|---|---:|---:|---:|---|
| SmolVLA (450M) | default: action expert + heads (100M trainable) | 8 | 3.6 GB | 0.26 | yes |
| SmolVLA | default | 32 | 8.8 GB | 0.97 | yes |
| SmolVLA | vision frozen, VLM + expert trained | 8 | 5.0 GB | 0.27 | yes |
| SmolVLA | full (vision + VLM + expert) | 2 | 6.7 GB | 0.26 | yes |
| SmolVLA | full | 4 | 9.4 GB | 0.39 | yes |
| SmolVLA | full, bf16 mixed precision, grad-accum ×4 (effective 16) | 4 | 8.6 GB | 0.18 | yes |
| SmolVLA | full | 8 | OOM | – | no |
| X-VLA (879M) | full, bf16 | 4 | 9.5 GB | 0.25 | yes |
| X-VLA | full, bf16 | 8 | OOM | – | no |
| X-VLA | VLM frozen, bf16 | 16 | 8.5 GB | 0.37 | yes |
| Evo-1 (0.77B, InternVL3-1B) | stage 1: action head only | 8 | 8.8 GB | 1.08 | yes |
| Evo-1 | stage 2: full, gradient checkpointing | 4 | 10.7 GB | 2.03 | yes (tight) |
| EO-1 (Qwen2.5-VL-3B) | full, bf16, grad ckpt | 2 | crashed at 7.8 GB (LeRobot integration bug) | – | unknown |
| π0.5 (4.14B) | action expert only, VLM frozen, bf16, grad ckpt | 1 | OOM | – | no |
| π0.5 | action expert only, same | 4 | OOM | – | no |
| π0.5 | LoRA r=16, bf16, grad ckpt | 1 | 9.4 GB | 0.29 | yes (batch 1 only) |

Not measured: GR00T N1.7 (extra deps) and MolmoAct2 (documented minimum above 12 GB); published
figures are used for those below. π0.5 expert-only runs out of memory because the expert (with
gradients and Adam state) is trained on top of 8.7 GiB of frozen bf16 weights; LoRA trains only small
adapters, so it fits.

Inference, AIC-shaped input (3 cameras 288×256, 26-dim state), 50-step chunk, nothing else on the GPU:

| Model | Process GPU memory | Warm latency per chunk | DSRL noise hook |
|---|---:|---:|---|
| SmolVLA | ~1 GB | 145–165 ms | works |
| π0.5 (`lerobot/pi05_base`, bf16, 4.14B) | 10.15 GiB | 208 ms | works (deterministic, steerable) |
| Evo-1 (published) | 2.3 GB | ~61 ms (16 Hz) | flow head |

Script: `scripts/bench_pi05_inference.py`. π0.5's checkpoint is a 14.5 GB fp32 download and needs an
HF login with Google's PaliGemma license accepted (gated tokenizer).

## 2. Candidate models

What matters here, in order: (1) fits a useful fine-tune on 12 GB; (2) a flow-matching/diffusion action
head, which DSRL needs (a regression head has no noise to steer); (3) a strong robot-data prior,
ideally single-arm end-effector control; (4) LeRobot support, so our server and dataset tooling carry
over; (5) inference speed for RL in the loop.

| Model | Size | Action head | Robot pretraining | 12 GB fine-tune | Notes |
|---|---|---|---|---|---|
| **X-VLA-0.9B** ([docs](https://huggingface.co/docs/lerobot/xvla)) | 0.9B, Florence-2 | flow matching | 290K episodes, 7 platforms (Droid, Robomind, Agibot) | full bf16 bs 4; frozen-VLM bs 16 | Soft prompts per embodiment; 9M-param prompt tuning reaches near-π0 on LIBERO/Simpler. `ee6d` / `auto` action modes suit Cartesian control. **Recommended.** |
| **Evo-1** ([docs](https://huggingface.co/docs/lerobot/evo1), [paper](https://arxiv.org/html/2511.04555v2)) | 0.77B, InternVL3-1B | flow matching (cross-modulated DiT) | none (relies on VLM alignment) | full bs 4 with grad ckpt | CVPR 2026; beats SmolVLA on Meta-World / RoboTwin, 94.8–96.7% LIBERO; 2.3 GB, 16 Hz. Needs more demos without a robot prior. Runner-up. |
| **SmolVLA** (current) | 0.45B, SmolVLM2 | flow matching | SO-100 community data (low-cost arms) | expert-only easily; full bs ≤ 4 | Already integrated and tested end to end. Weakest prior. Keep as baseline. |
| VLA-Adapter ([paper](https://arxiv.org/pdf/2509.09372)) | 0.5B, Qwen2.5 | regression | none | ~10 GB | Strong LIBERO, but no noise → no DSRL. Residual RL only. |
| π0.5 ([docs](https://huggingface.co/docs/lerobot/pi05)) | 4.14B, PaliGemma | flow matching | large, diverse | LoRA batch 1 only (9.4 GB); expert-only OOM | Best RL ecosystem (DSRL, πRL, eRLT, EXPO-FT) and the contact-rich study below. Inference 10.15 GiB leaves no room for Gazebo + RL. **24 GB option.** |
| MolmoAct2 ([docs](https://huggingface.co/docs/lerobot/main/en/molmoact2)) | large, Molmo | flow matching + tokens | yes | no (16.5 GiB expert-only, 20.2 GiB LoRA) | Apache 2.0; 98% LIBERO. 24 GB option. |
| GR00T N1.7 | 3B | flow DiT | humanoid-heavy | no (default ~35 GB; LoRA on 24 GB) | Not a good fit. |

**About the 24 GB GPU.** DSRL and residual RL keep the VLA frozen, so RL needs only *inference*
memory. Measured: π0.5 inference alone takes 10.15 GiB of the 11.49 GiB usable. Headless Gazebo
needs ~1 GB (2.5 GB with the observer cameras), plus the RL actor, critic and replay batches, so RL
with π0.5 in the loop does not fit on 12 GB in bf16 (8-bit weight quantization might, untested).
LoRA fine-tuning does fit locally at batch 1 (use gradient accumulation for an effective batch).
So π0.5 is a 24 GB plan: fine-tune and run RL on the borrowed card. On 12 GB, X-VLA remains the pick.

**Whatever the model, three things matter as much as which one:**
1. Put the wrist force/torque (and a short history) in the state vector. The AIC recorder's 26-dim
   state has no wrench. A study of π0/π0.5 on plug, USB and key insertion
   ([FACT](https://arxiv.org/html/2608.01402)) found force history worth 40–60 points of success and
   that VLAs shrink their actions to near zero at contact ("delta collapse").
2. Use a low-noise-heavy flow schedule (logit-normal) when fine-tuning a flow VLA for contact; FACT
   reports up to +20 points on precision tasks from that alone.
3. Camera resolution: AIC images are scaled 0.25 (288×256). Fine for approach, likely too coarse for
   sub-millimetre alignment, which is where RL with force has to take over.

## 3. Is imitation learning reinforcement learning?

No. Behavior cloning is supervised learning: regress the expert's action given the observation. There
is no reward, no exploration and no notion of improving beyond the demonstrator, and errors compound
once the policy drifts off the demonstrated states. DAgger adds interaction but is still imitation.
Offline RL uses rewards but no new interaction. **Online RL** (the policy acts, gets rewards, improves
from its own experience) is what the course is about.

In modern robot learning, BC usually provides the *prior* and RL does the improving. That is the right
framing here: a pretrained VLA plus a small demo set give the RL agent something sensible to explore
around, because RL from scratch on a sparse insertion reward rarely finds the hole.

## 4. Is DSRL the right RL method?

[DSRL](https://arxiv.org/abs/2506.15799) trains a small SAC agent whose action is the initial noise of
the frozen diffusion/flow policy. It is genuine online RL, tiny (~1 GB), sample-efficient, needs only
black-box access to the VLA, and stays safe because every action comes from the base policy.

**Its limitation is exactly our hard part.** DSRL can only choose among behaviors the base policy can
already produce. EXPO-FT's authors put it plainly: DSRL's optimization is "constrained to modes
already covered by the offline data." For insertion, the base VLA's failure is a precision and
contact failure (sub-millimetre misalignment, no force reaction), and picking a different noise sample
cannot add a correction the base never makes. In real precision-insertion benchmarks, methods that
can step outside the base distribution beat it: [eRLT](https://arxiv.org/html/2610.00913v1) (frozen
π0.5 + actor-critic) beat DSRL and took USB insertion from 50% to 90% in ~80 trajectories (~20–25 min,
one Blackwell GPU); [EXPO-FT](https://arxiv.org/html/2605.25477) reached 30/30 in ~19 min of robot
data with bounded "edit" actions.

| Approach | What is trained | Can exceed base behavior? | 12 GB feasible? | Verdict |
|---|---|---|---|---|
| DSRL ([paper](https://arxiv.org/abs/2506.15799)) | noise policy (SAC) | no | yes (~1 GB) | Keep, as one arm of the comparison |
| Residual off-policy RL ([ResFiT](https://arxiv.org/pdf/2509.19301), [ForceRFT](https://arxiv.org/pdf/2609.22840)) | bounded per-step correction (SAC) on frozen base | yes | yes | **Primary** |
| Frozen-VLA features + actor-critic ([eRLT](https://arxiv.org/html/2610.00913v1)) | small actor/critic on VLA tokens | yes | yes | Use its idea: feed VLA features to actor/critic |
| Edit policy + full VLA update ([EXPO-FT](https://arxiv.org/html/2605.25477)) | edit policy + VLA | yes | no (2×H200) | Borrow the bounded-edit idea only |
| Full VLA PPO/GRPO ([πRL](https://rlinf.readthedocs.io/en/release-v0.2/rst_source/publications/pi_rl.html), [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)) | whole VLA | yes | no (8×A800 80 GB) | Out of reach |
| HIL-SERL / RLPD (in LeRobot) | small policy from scratch + demos in buffer | yes | yes | Good for a non-VLA baseline |

### Recommended design: "steer + correct" on a frozen VLA

```
obs (3 cams, proprio, F/T history) ──▶ frozen VLA ──(noise z from actor)──▶ action chunk a_vla
                                          │ features
actor  π(z, Δa | VLA features, proprio, F/T history)      Δa bounded, e.g. |Δv| ≤ 1 cm/s
command = a_vla[t] + Δa[t]   (20 Hz)                       z re-chosen each chunk (DSRL)
critic Q(s_priv, z, Δa)   s_priv adds ground-truth plug/port pose (sim only, asymmetric)
```
- **Algorithm:** SAC in RLPD style: high update-to-data ratio, critic ensemble with layer norm, and
  CheatCode demos mixed 50/50 into replay batches. All online, off-policy and sample-efficient.
- **Reward (training only, from Gazebo ground-truth TF with `ground_truth:=true`):** dense terms that
  mirror the AIC scorer: −distance(plug, port), + insertion depth, a large success bonus, − force
  above 20 N, − off-limit contacts, − jerk. The deployed policy never sees ground truth.
- **Where to train:** directly in Gazebo, the evaluator itself, so there is no sim-to-sim gap. It
  runs at ~1× real time after this week's GPU-rendering fix, giving roughly 60–100 trials per hour,
  in line with the 80–300 episode budgets above (overnight runs). Isaac Lab (`AIC-Task-v0`, parallel
  envs, ~41 env-steps/s with cameras at 4 envs) is for scaling or pretraining the residual on state
  only, with the sim-to-sim gap as the cost.
- **Course experiment:** BC-only VLA vs + DSRL vs + residual vs + both, plus a classical spiral-search
  baseline. Report learning curves, success, force-penalty counts and AIC score.

Context from the competition itself: the [winners](https://www.intrinsic.ai/blog/posts/ai-for-industry-challenge)
used a perception foundation model for port pose, then classical force-controlled insertion (spiral
search, alignment correction). 90% of the top 10 went classical for the insertion step. A
force-aware RL residual is the learned counterpart of that search behavior, which is a good story for
an RL course and a strong baseline to beat.

## 5. Why fine-tuning uses so much VRAM, and what can be cut

What is in GPU memory during one training step:

| Item | Persists across batches? | Why | Size (Adam, fp32) |
|---|---|---|---|
| Weights | yes | it's the model | 4 B / param (2 in bf16) |
| Gradients | buffer reused each step | computed every backward, then consumed by the optimizer | 4 B / trainable param |
| Optimizer state | **yes, by algorithm** | Adam keeps running averages of the gradient and its square; they *are* the optimizer's memory of past batches | 8 B / trainable param |
| Activations | **no** | saved during forward for use in backward, freed when backward ends | grows with batch size and image tokens |

So nothing from batch *k*'s data stays on the GPU for batch *k+1*; activations are freed. What stays
is the model, its optimizer state and gradient buffers. Two things make it look otherwise:
- PyTorch's caching allocator keeps freed blocks reserved for reuse (much faster than returning them
  to the driver), so `nvidia-smi` shows the peak, not current use.
- The peak is set by the single largest step, which is forward + backward of one batch.

Measured on SmolVLA: per-sample activation cost is about 0.22 GB when only the action expert trains,
and about 1.3 GB when the vision encoder trains too, because then the image activations must be kept
for backward. That is why full fine-tuning OOMs at batch 8 while expert-only runs batch 32.

What cuts it (✓ = measured here):
- ✓ **Freeze parts:** no gradients, no optimizer state, no saved activations for them. Full SmolVLA
  OOMs at batch 8; with vision frozen it's 5.0 GB.
- ✓ **Small micro-batch + gradient accumulation:** mathematically the same update as a large batch.
  SmolVLA full, 4 × 4 accumulation (effective 16) at 8.6 GB.
- ✓ **bf16 mixed precision:** halves activations (part of the 8.6 GB above, and 2× faster).
- ✓ **Gradient checkpointing:** don't store activations; recompute them during backward (~20–30% more
  compute). This is how Evo-1's full fine-tune fits.
- ✓ **Allocator tuning** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): less fragmentation,
  −0.2 GB.
- **8-bit / paged AdamW** (bitsandbytes): optimizer state 8 → ~2 B/param.
- **LoRA:** gradients and optimizer state only for small adapters.
- **Optimizer CPU offload** (DeepSpeed ZeRO-Offload, FSDP offload): keep Adam state in system RAM
  (you have 32 GB), at the cost of PCIe transfer time.
- For RL with a frozen VLA, none of this applies: it's inference only (~1 GB for SmolVLA).

## 6. Next steps
1. Add wrist wrench (+ short history) to the recorder's state and to `policy/aic_smolvla/contract.py`.
2. Record ~200–300 CheatCode demos automatically (no teleop) as a LeRobot dataset.
3. Fine-tune X-VLA (full bf16, bs 4) and SmolVLA (bs 4, bf16, accum 4) on it; compare in Gazebo.
4. Build the SAC "steer + correct" agent on the existing server (its `noise` hook is already there),
   with a Gazebo reward node reading ground-truth TF.
5. Run the four-way comparison.
