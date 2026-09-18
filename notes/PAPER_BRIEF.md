# Paper brief

This file is the durable shared understanding between authors and agents. Fill it conversationally; short bullets are enough. Do not treat unknown fields as permission to invent details.

## Identity

- Working title: Do Language Models Recognize a Phrase Before Reading It?
- Target venue and deadline: Amirim final project (HUJI, CS 68101). Written in ACL format so it can be extended toward an *ACL submission. No external deadline fixed.
- Paper type: empirical study / analysis (with a small methodological contribution)
- Intended reader: an interpretability-literate NLP reader who knows Patchscopes and the logit lens, and who cares whether "the model already knows X" claims survive a change of instrument.

## One-sentence paper

When a language model is partway through a familiar multi-word expression, does its hidden state already encode the rest — and how much does that answer depend on the tool used to read the state out?

## Main contributions

1. A **lookahead-distance protocol** for fixed multi-word expressions: stop the model `i` tokens before the end of a known phrase and ask whether the remaining `i` tokens are recoverable. Applied to 1,164 phrases (idioms, landmarks, film titles) in isolation and in 9,955 naturally occurring FineWeb-Edu contexts.
2. **Evidence that phrase-level lookahead is real and is a property of the phrase, not of the occasion.** Per-phrase success at `i=3` is strongly bimodal: of 784 phrases with ten contexts each, 40.8% succeed in 9–10 of 10 and 22.3% in 0–2 of 10; only 19.4% sit in the middle. A per-context noise process would concentrate in the middle.
3. **Evidence that the readout instrument changes the conclusion.** Patchscopes and ordinary greedy generation, run on the *identical* 24,740 (phrase, context, `i`) instances, agree on only 81.1% of them and disagree in both directions (1,924 patchscopes-only vs 2,751 generation-only). They also rank the three phrase categories differently.
4. **A hidden-state probe that predicts whether the model will actually complete the phrase**, at 76.8% balanced accuracy and 87.6% precision (L30), and well above chance already at L1 — with the caveat that ~80% of its apparent false positives are valid alternate phrasings rather than real errors.

## Editorial center

- **The one finding readers should remember:** the hidden state mid-phrase carries a usable, phrase-stable signal about what is coming — but "how much the model knows" is not instrument-independent, and Patchscopes systematically measures something different from what the model actually does.
- **The tension that makes it matter:** the interpretability literature routinely treats a successful Patchscopes readout as evidence that "the model knows X." We run Patchscopes and the model's own behaviour side by side on identical inputs and find they disagree ~19% of the time, in both directions, and that the disagreement is not random — it flips a category ranking and moves the apparent "best layer."
- **Evidence essential to that conclusion:** the matched-instance agreement table; the per-phrase bimodality; the two probes (patchscopes-label vs generation-label) and how their layer profiles differ; the false-positive adjudication.
- **Useful but secondary (appendix):** the OLMo-3 and Qwen2.5-14B replications; the 5-slot patchscope prompt ablation; per-layer success curves; held-out-category probe generalization; model-confidence analysis.
- **Project history that should not appear:** the BOS-token bug, the generation-budget bug, the cluster setup, the earlier "does it recognize at *any* position" framing. These shaped the work but are not results. The buffer fix *is* worth one methods sentence, because it is why `i>=6` is no longer reported as zero.

## Claims and evidence

| Claim | Evidence that would support it | Current evidence/location | Scope or caveat |
|---|---|---|---|
| Models complete familiar phrases well before the final token | Success rate at `i>=2` far above chance | `results/processed/paper_numbers.json` → `in_context.generation_olmo2`: 84.0% at `i=2`, 64.8% at `i=3`, 55.6% at `i=4`, 53.6% at `i=5` | Success = the remaining token string appears in the continuation; lenient about trailing text, strict about wording |
| Natural context substantially raises recognition | Same phrases, isolated vs in context | `isolated.generation_olmo2` (59.2 / 31.2 / 20.6 / 17.6 at `i=2..5`) vs `in_context.generation_olmo2` (84.0 / 64.8 / 55.6 / 53.6) | Isolated phrases are a lower bound, not the natural operating condition |
| Recognition is a stable property of the phrase | Bimodal per-phrase success distribution | `per_phrase_consistency_i3` | Computed at `i=3` on the 784 phrases with exactly 10 contexts |
| Patchscopes and real generation measure different things | Matched-instance disagreement in both directions | `agreement_32L_pooled_i2_5`: 81.1% agreement, 1,924 patchscopes-only, 2,751 generation-only | Full 32-layer patchscopes sweep. The 8-layer subset used for probing agrees only 78.7% and understates patchscopes badly |
| Category ranking depends on the instrument | Same categories, two readouts | `category_pooled_i2_5` | Under generation: movie 84.5 > building 75.3 > idiom 66.4. Under patchscopes(32L): movie 68.6 > idiom 65.9 > building 56.8. The ranking changes; the "idioms best → worst" flip only holds for the 8-layer subset |
| The hidden state predicts real completion success | Probe trained on hidden states, generation label | `external.generation_label_probe`: 76.8% bal. acc., 87.6% precision at L30 | Not recomputable from this repo; feature arrays live on the cluster |
| The signal is learned structure, not base rate | Probe at L0/L1/L2 | `external.generation_label_probe.early_layers_*`: 58.6% bal. acc. at L0 rising to 76.8% at L30 | L0 is the raw embedding and is near chance for the patchscopes label (49.5%) |
| Strict string match understates real knowledge | LLM adjudication of all false positives | `external.false_positive_llm_judgement`: 62.9% valid alternate, 17.5% partial, 19.7% unrelated | Single-judge, unvalidated rubric. Report as indicative, not as a measured accuracy correction |
| The effect is not specific to one model | Replication on OLMo-3-7B and Qwen2.5-14B | `in_context.patchscopes_olmo3_32L`, `external.qwen25_14b_generation_pct` | Qwen2.5-14B is ~2x the parameters and does *not* clearly win. Report this; do not bury it |

## Story and structure

- **Starting point the reader likely accepts:** decoder-only LMs are trained to predict one token at a time, and recent work shows they nonetheless build unified representations of units larger than a token (Kaplan et al., 2025, for words).
- **Concrete problem or motivating example:** at the token `Empire` in "the Empire State Building", the model must still run two more full forward passes. If `State Building` is already latent at `Empire`, those passes are partly redundant.
- **Gap or question:** the word-level result is about a unit the model has *already finished reading*. Nothing establishes that the model represents material it has **not yet read**. That is a different claim, and it is the one with efficiency consequences.
- **Core idea or answer:** it does, for familiar fixed expressions, up to a few tokens ahead — but the size of the effect is instrument-dependent, and the instrument most used in this literature is not the one that tracks behaviour best.
- **Atomic claim chain:** phrases are recognized ahead of time (A) + recognition is phrase-stable, not contextual noise (B) → the hidden state carries a reusable phrase-level signal (C); C + the signal is linearly decodable and predicts real behaviour (D) → a cheap extractor is plausible (E); but A–D are measured differently by two instruments that disagree (F) → claims of the form "the model knows X" must name their readout (G).
- **Final takeaway:** phrase-level lookahead is real, stable, and decodable; and interpretability claims about it are only as strong as the readout that produced them.

## Terminology

| Concept | Preferred term | Avoid / distinguish from |
|---|---|---|
| Number of phrase tokens still unread | **lookahead distance `i`** | "position", "offset" |
| Reading a hidden state via an injected prompt | **Patchscopes readout** | "the model knows" without qualification |
| Greedy continuation from the true prefix | **generation readout** | "ground truth" (it is a second instrument, not truth) |
| A phrase recovered at some layer | **recovered / recognized at distance `i`** | "predicted" (suggests next-token prediction only) |
| Fixed multi-word expression | **phrase** | "entity", "collocation" |

## Open decisions

- **Which patchscopes number is the headline?** The August deck reports the 8-layer union (73.2/57.5/43.9/38.7 at `i=2..5`) and concludes generation beats patchscopes by 7–15 points. The full 32-layer sweep exists in `results/raw/lookahead_with_context_full.json` and gives 79.6/64.1/51.6/46.4, shrinking the gap to 0.7–7.2 points and erasing it almost entirely at `i=3`. **The draft leads with the 32-layer comparison** as the fair one and reports the 8-layer subset where the probe requires it. Authors should confirm.
- The "idioms flip from best to worst" framing in the deck does not survive the 32-layer sweep. The draft makes the weaker, still-supported claim: the ranking is instrument-dependent.
- Tokenization means the `i`-token target sometimes starts mid-word (`istine Chapel`, `ulp Fiction`). Currently reported as a limitation. Decide whether to re-run with word-aligned targets.
- The false-positive adjudication used a single LLM judge with no human validation. Decide whether to validate a sample before the number stays in the paper.
- Probe metrics are transcribed from the July reports because the hidden-state arrays are on the cluster. Decide whether to re-run and archive them for the artifact.

## Interview status

- Last interview/checkpoint: 2026-09-18 — brief reconstructed by agent from the August 2026 summary deck, the July 2026 analysis reports, and the raw result files, not from a live author interview.
- Questions still requiring author judgment: everything under Open decisions; the target venue; and whether the efficiency motivation (skip-ahead decoding) should stay as framing given that no decoding experiment is reported here.
