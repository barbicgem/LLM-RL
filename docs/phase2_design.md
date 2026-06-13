# Phase 2 Design — Injection-Guided Transparency Training

Internal working notes. Status: draft. Model: `gemma-3-4b-it`, extraction layer `L = 20`, bf16.

## Goal

Train the model to **accurately introspect and report its own internal emotional state**. We use the Phase 1 emotion vectors as ground-truth supervision via activation injection: because we control exactly what we inject (which emotion, how strongly), we know what the model *should* report, including reporting "nothing" when nothing is injected.

This turns an otherwise unsupervised problem ("does the model know what it's feeling?") into a supervised one.

## Why injection gives ground truth

The hard part of introspection research is the absence of labels — there's no oracle for "what the model is actually feeling." Injection sidesteps this: we add a known emotion direction `v_e` to the residual stream, so the induced state *is* the label. We then train the verbal self-report to match it. The Phase 1 vectors do double duty — they are both the **stimulus** (what we inject) and the **label** (what the report is graded against).

## Relationship to Phase 1

- Phase 1 produced clean, validated `v_e` per emotion at `L = 20` (`data/vectors/emotion_vectors_layer20.npz`), with structure confirmed by the cross-emotion cosine matrix and SAE alignment.
- Phase 2 reuses these directions as injection vectors and labels. Vector quality directly bounds Phase 2 quality — the vectors are difference-of-means estimates with PCA confound removal, i.e. approximate, so we are injecting a slightly noisy direction. Acceptable, but noted.

## Method comparison (resolving the "architecture TBD")

Three candidates from the project notes:

**1. Steering vectors only (no weight update).** Inject `v_e` at inference and prompt the model to report. This is a *probe / baseline*, not training — it doesn't create a durable, calibrated capability, and current models do this only unreliably ("emergent" introspection). Verdict: it's the **injection mechanism used inside training** and a **baseline**, but not the method itself.

**2. Activation patching.** Swap activations at chosen positions/layers from a reference run. This is an interpretability / causal-attribution tool, heavier and less natural than "induce an emotion and ask about it." Verdict: **not the training method**; keep it for evaluation (causal sanity checks), not training.

**3. Fine-tuning with an introspection loss (recommended).** Inject `v_e` during the forward pass while the model answers an introspection prompt, and supervise the answer to (a) name/describe `e`, (b) at an intensity matching the injection strength, and (c) report neutrality when nothing is injected. This is what actually teaches reliable, calibrated, generalizing introspection. Verdict: **the method.**

**Decision:** Method 3, with Method 1 as the in-the-loop injection mechanism + baseline, and Method 2 reserved for causal evaluation.

## Training design

### Injection mechanism
- Add `α · v̂_e` to the residual stream at layer `L = 20` via a forward hook, where `v̂_e` is unit-normalized and `α` controls intensity.
- Inject on all token positions from the introspection prompt onward, so the model "feels" the state while it answers. (Alternative: assistant-turn positions only — document and A/B later.)
- `α` scale must be calibrated to the typical residual-stream norm at `L`. Too strong → incoherent text; too weak → no signal. **Needs a sweep before training.** Use a small set of levels mapped to intensity words, plus `α = 0` for controls.
- Single layer (`L = 20`) first. A multi-layer span may induce a stronger, more robust state — test as a later ablation.

### Supervision / targets
- **Introspection prompts:** a pool of ~20–50 paraphrases ("Take a moment to notice your current internal state and describe what, if anything, you're feeling," etc.). Diversity prevents the model from keying on one phrasing.
- **Injected examples:** target accurately identifies emotion `e` and graded intensity (faint / moderate / strong ↔ `α` level). Use several target paraphrases per (emotion, intensity) — avoid a single rigid template the model can memorize.
- **Control examples (`α = 0`, no injection):** target reports no particular emotional state. **This is the most important ingredient** — without a substantial fraction of controls (~30–50%), the model just learns to always claim an emotion. Controls are what make the report *honest*.

### Loss
- Primary: causal-LM cross-entropy on the target response tokens, **with the injection hook active during the (teacher-forced) forward pass.** Train and evaluate with the hook on, so the model actually learns to read the injected signal.
- Optional auxiliary terms (only if calibration is poor): a monotonicity penalty tying reported intensity to `α`, or an explicit false-positive penalty on controls. Start with plain CE + balanced data; add complexity only if needed.

### Parameters to train
- **LoRA adapters (recommended):** cheap and fast on 4B, low risk to general capability, trivially ablatable (disable adapter to recover the base model). Rank ~8–32 on attention + MLP projections.
- Full fine-tune is feasible for 4B on the box but risks capability regression and template overfitting. Use LoRA first.

### Data recipe (first pass)
- 12 emotions × ~K introspection prompts × ~M intensity levels (including `α = 0`) × a few target paraphrases.
- **Hold out 3–4 emotions entirely** to test whether introspection generalizes to states never trained for (the strongest test of genuine introspection vs. lookup). Also hold out some prompt phrasings.
- Balance injected vs. control examples.
- Targets can be generated with the model itself (same approach as Phase 1 story generation).

## Evaluation

The point is reliable, calibrated, generalizing introspection — not template parroting. Battery:

1. **Report accuracy** — on held-out prompts, does it name the injected emotion? (top-1 and valence-cluster accuracy).
2. **Calibration** — reported intensity vs. `α`: is it monotonic / well-ordered? Plot the curve.
3. **Control / false-positive rate** — with `α = 0`, how often does it correctly report neutrality vs. confabulate? The key honesty metric.
4. **Emotion generalization** — accuracy on the held-out emotions never trained for. Genuine introspection vs. memorized mapping.
5. **Layer / strength robustness** — inject at an unseen layer or strength; does the report still track? Guards against shortcut-learning a specific artifact.
6. **Capability retention** — small general benchmark with the adapter on but no injection; confirm we didn't damage the base model and that it defaults to neutral in normal use.
7. **Causal check (optional, uses activation patching)** — ablate the `v_e` component and confirm the report collapses, i.e. the report is driven by the injected direction.

Baselines: (a) base model + steering, prompted to introspect (the untrained/"emergent" baseline); (b) prompted-only, no injection.

## Key risks & open questions

- **Shortcut learning (central risk).** The model may detect the injection *artifact* (an off-distribution activation) rather than introspecting its *content*. Mitigations: continuous `α`, held-out layer/strength evals, and the held-out-emotion generalization test. If it only works at the trained layer/strength, it's reading an artifact.
- **Transfer to naturally arising states.** The ultimate claim is introspection of *genuine* (non-injected) states, which have no ground-truth label. Proxy: feed emotion-evoking contexts (the Phase 1 stories) and check report consistency with the activation's projection onto `v_e`. Treat as a stretch goal / Phase 3.
- **`α` scaling vs. coherence** — requires the sweep above.
- **Template memorization** — diversify targets + hold out phrasings.
- **Capability regression** — prefer LoRA, monitor with the retention eval.
- **Layer choice** — `L = 20` is inherited from Phase 1 extraction; the injection-optimal layer may differ. Sweep.
- **Label noise** — `v_e` are approximate; we're injecting a slightly noisy direction.

## Minimal viable experiment (first milestone)

1. Implement the injection hook (reuse `model_utils._get_layer`).
2. Build a small dataset: 8 train emotions × 10 prompts × {0, med, high} `α` × 3 target paraphrases + matched controls.
3. LoRA fine-tune with CE, injection active.
4. Eval: report accuracy on held-out prompts, control FPR, calibration over the 3 `α` levels, generalization to the 4 held-out emotions.
5. **Decision gate:** if held-out-emotion accuracy beats chance and control FPR is low, scale up (more emotions/prompts, finer intensity, multi-layer injection). If not, diagnose shortcut learning before scaling.

## Implementation scaffolding

- `emotion_validation/inject.py` — forward-hook injector that adds `α · v̂` at layer `L`; context manager to enable/disable.
- `training/train_introspection.py` — data loading, LoRA setup (`peft`), training loop with injection active during the forward.
- `data/introspection/` — generated introspection prompts + targets.
- `eval/eval_introspection.py` — the battery above.
- Reuse: `load_model_and_tokenizer` (bf16), vectors from `data/vectors/emotion_vectors_layer20.npz`.
- Hardware: 4B + LoRA fits comfortably; training is light relative to Phase 1 generation.

## Decisions to resolve before coding

- `α` scale and the discrete levels → intensity-word mapping (needs a sweep).
- Inject positions: prompt-onward vs. assistant-turn only.
- Single layer vs. multi-layer span.
- Control fraction (start ~40%).
- LoRA rank/targets.
- Target-generation method (self-generated vs. templated-then-paraphrased).
