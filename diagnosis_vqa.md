# VQA CoT Diagnosis — why stage-2 reasoning training made the model worse

Model: `Qwen3.5-0.8B` → +SFT LoRA → `qwen35-vqa-merged` → +CoT LoRA → `qwen35-cot-vqa-merged`
Date: 2026-07-14

---

## TL;DR

CoT training cost **8.3 points of mean accuracy** on RSVQA, and it regressed on *every*
dataset. This is not a hyperparameter problem. Two independent findings explain it:

1. **The CoT stage adds zero new information.** The SFT and CoT datasets contain the
   *identical* 5,380 (image, question) pairs — 100% overlap. Only the target text
   differs. So stage-2 cannot teach the model to perceive anything new; it can only
   overwrite the direct-answer pathway with a lossier reason-then-answer pathway.
2. **The failures are perceptual, and CoT makes perception worse.** 140 of 186 wrong
   answers are cases where the *reasoning itself* is wrong — the model fails to see the
   object, verbalises that miss, and the `<answer>` faithfully follows the bad reasoning.

**Recommendation: ship the stage-1 SFT model for VQA. If CoT is wanted, train on a
mixture of reasoning and direct-answer targets instead of 100% reasoning.**

---

## Results

| Dataset | Qwen2VL-2B (ref) | Stage-1 SFT | Stage-2 CoT | Δ (CoT − SFT) |
|---|---|---|---|---|
| RSVQA LR-rural_urban | 93.0% | **92.0%** | 80.0% | −12.0 |
| RSVQA LR-presence | 91.3% | **89.5%** | 80.4% | −9.1 |
| RSVQA LR-comp | 90.4% | **90.1%** | 82.2% | −7.9 |
| RSVQA HR-comp | 87.6% | **79.9%** | 72.9% | −7.0 |
| RSVQA HR-presence | 57.8% | **54.4%** | 49.2% | −5.2 |
| **Mean** | **84.0%** | **81.2%** | **72.9%** | **−8.3** |

Stage-1 SFT is a good result on its own: within **2.8 points** of Qwen2-VL-2B with a model
**2.5× smaller**. Stage-2 is a uniform regression.

CoT training converged normally (loss 2.02 → 0.43 over 211 steps, 1 epoch), so this is not
undertraining.

---

## Root cause 1 — the CoT stage sees no new data

`datasets/VQA_sft/vqa_sft.json` and `datasets/reasoning/VQA/vqa_sft_reasoning.json` are
built from the same source rows (`list_sft_oversampled.json` vs
`list_sft_reasoning_oversampled.json`):

```
SFT entries=26900   CoT entries=26900
unique images                : SFT=5377  CoT=5377  shared=5377  (100.0%)
unique (image, question)     : SFT=5380  CoT=5380  shared=5380  (100.0%)
CoT (image,question) pairs NOT seen in SFT: 0
```

The only difference is the assistant target:

- SFT: `no`
- CoT: `<reasoning>…</reasoning>\n<answer>no</answer>`

Consequences:

- Stage-2 **cannot improve perception** — no new images, no new questions. The model
  already fit these pairs to convergence in stage-1.
- The only thing stage-2 changes is the *output mapping* on inputs it has already learned:
  it replaces a working direct-answer pathway with a reason-then-answer pathway.
- The reasoning traces are **synthetic**, distilled from a larger model whose perceptions a
  0.8B model does not share. So the replacement pathway is noisier than the one it removed.

No amount of rank / LR / epoch tuning creates information that is not in the data.

---

## Root cause 2 — CoT converts a soft perceptual signal into a hard wrong commitment

Breakdown of the 186 wrong CoT predictions on RSVQA LR-presence:

| | count |
|---|---|
| Reasoning itself reached the **wrong** conclusion | **140** |
| Reasoning correct but `<answer>` disagreed with it | 6 |
| Ambiguous | 40 |

The answer head is fine — it faithfully reports whatever the reasoning concluded. **The
reasoning is what breaks.** Example (gold = `yes`, a circular building *is* present):

> "…the roads and paths are straight or slightly curved but do not form any circular
> structures. Given this observation, I conclude that there is **no** circular building
> visible in the image."

The model never saw the building, verbalised that miss, and the `<answer>` followed.

**The mechanism.** RSVQA questions are perceptual lookups ("is a circular building
present?"), not multi-step problems. In SFT the answer is read straight off the visual
features — a weak, holistic "something roundish is there" signal is enough to emit `yes`.
In CoT the model must first *name* what it sees in prose; if it fails to name the object,
the reasoning concludes absence, and `<answer>` is conditioned on that prose. A soft signal
that would have produced a correct `yes` becomes an explicit "I don't see it" → `no`.

This shows up as a systematic **false-negative drift**:

| LR-presence | gold | SFT pred | CoT pred |
|---|---|---|---|
| `yes` | 706 | 664 | 624 |
| `no` | 244 | 286 | **326** |

And CoT breaks far more than it fixes:

| Dataset | SFT right → CoT **wrong** | SFT wrong → CoT **right** |
|---|---|---|
| HR-comp | 142 | 72 |
| LR-presence | 125 | 39 |

Reasoning does genuinely rescue some samples (39–72 per set) — it just destroys 2–3× more.

### Secondary: noisy traces

5.2% of yes/no training traces reason to a conclusion that **contradicts their own gold
answer** (1,330 / 25,525). That teaches incoherence — "produce this reasoning, then ignore
it" — though it is too small to explain the full 8.3-point drop on its own.

---

## Fixes, in order of expected payoff

1. **Mix reasoning and direct-answer targets (do not train on 100% reasoning).**
   Qwen3.5 supports this natively: with `--enable_reasoning True` the `reasoning` field is
   *optional per sample* (see README — Qwen3.5 is "the only supported family where samples
   may mix reasoning and non-reasoning data"). Samples without it train the direct answer
   under the closed-`<think>` scaffold; samples with it train reasoning inside the native
   `<think>` block. The direct half acts as replay and stops the SFT skill from being
   overwritten. Both target variants already exist over identical pairs, so the mix is
   trivial to build. Suggested start: 50/50.

2. **Filter the contradictory traces** (the 5.2% above). Cheap, small gain.

3. **Reserve CoT for tasks where reasoning has something to do.** RSVQA is perceptual
   lookup. VG ("the tennis court *left of* the golf field") is genuinely compositional and
   is the better candidate for a CoT win. Do not assume the VQA result transfers.

4. **For VQA today: ship the stage-1 SFT model** (`qwen35-vqa-merged`, 81.2% mean).

---

## Appendix — eval bug found along the way (fixed)

The first CoT eval reported **0.0% accuracy on every dataset**. This was *not* a model
failure. The eval passed `--template_config` (the TinyRS `qwen2_thinking_template.json`
system prompt), but the CoT models were trained with **no system turn at all** (Qwen3.5
gets no default system message — `use_default_system_message()` is true only for
`qwen2_vl`/`qwen2_5_vl`). That unseen ~90-token prefix pushed the model onto a degenerate
greedy path.

Measured A/B on `qwen35-cot-vqa-merged`, same samples, same greedy decoding:

| | well-formed output |
|---|---|
| **With** system prompt | **0 / 6** |
| **Without** system prompt | **6 / 6** |

With the system prompt the model emits `<reasoning>` + prose and then closes with
`</answer>`, skipping `</reasoning>`, `<answer>`, and the answer word itself. (Both closing
tags share the leading `</` token; after prose ending "…the answer is no.", greedy picks
`answer` — closing a block it was never in.) Nothing is parseable → every sample scores
false → 0.0.

Two fixes applied:

- Removed `--template_config` from the CoT eval scripts. The tag format is baked in by
  training; the instruction is unnecessary *and* destructive.
- `clean_prediction()` in `eval_cls_vllm.py` previously swallowed the parse failure with a
  bare `except: pass`, leaving the whole reasoning paragraph as the "prediction" — so an
  unparseable model silently read as 0.0 accuracy instead of an error. It now returns a
  `malformed` flag, records `malformed_outputs` in the metric JSON, and prints a warning
  that the accuracy is not a model-quality signal.

**Rule of thumb: evaluate with the same prompt structure the model was trained on. If
`malformed_outputs` is non-zero, suspect the prompt before suspecting the model.**
