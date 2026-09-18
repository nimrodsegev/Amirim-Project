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
2. **Evidence that phrase-level lookahead is real and is a property of the phrase, not of the occasion.** Over the 803 phrases with ≥5 *distinct* contexts, per-phrase success at `i=3` is far more bimodal than an independent per-context process: 31.5% succeed in ≥90% of their contexts (binomial null 10.1%) and 18.3% in ≤20% (null 5.0%), while only 27.3% fall in the 40–60% band that holds 48.5% of the null's mass.
3. **Evidence that the readout instrument changes the conclusion.** Patchscopes and ordinary greedy generation, run on the *identical* 24,740 (phrase, context, `i`) instances, agree on only 78.7% of them, with generation recovering what patchscopes misses ~3× as often as the reverse. They reverse the ranking of the three phrase categories. Widening the patchscopes sweep to all 32 layers accounts for much of the asymmetry (agreement 81.1%, ratio ~3:2) but not for the disagreement itself.
4. **A hidden-state probe that predicts whether the model will actually complete the phrase**, at 76.8% balanced accuracy and 87.6% precision (L30), and well above chance already at L1 — with the caveat that ~80% of its apparent false positives are valid alternate phrasings rather than real errors.

## Editorial center

- **The one finding readers should remember:** the hidden state mid-phrase carries a usable, phrase-stable signal about what is coming — but "how much the model knows" is not instrument-independent, and Patchscopes systematically measures something different from what the model actually does.
- **The tension that makes it matter:** the interpretability literature routinely treats a successful Patchscopes readout as evidence that "the model knows X." We run Patchscopes and the model's own behaviour side by side on identical inputs and find they disagree ~21% of the time, that the disagreement is not random — it reverses a category ranking and moves the apparent "best layer" — and that a large part of its apparent size is set by a methodological choice (how many layers the readout unions over) rather than by the model.
- **Evidence essential to that conclusion:** the matched-instance agreement table; the per-phrase bimodality; the two probes (patchscopes-label vs generation-label) and how their layer profiles differ; the false-positive adjudication.
- **Useful but secondary (appendix):** the OLMo-3 and Qwen2.5-14B replications; the 5-slot patchscope prompt ablation; per-layer success curves; held-out-category probe generalization; model-confidence analysis.
- **Project history that should not appear:** the BOS-token bug, the generation-budget bug, the cluster setup, the earlier "does it recognize at *any* position" framing. These shaped the work but are not results. The buffer fix *is* worth one methods sentence, because it is why `i>=6` is no longer reported as zero.

## Claims and evidence

| Claim | Evidence that would support it | Current evidence/location | Scope or caveat |
|---|---|---|---|
| Models complete familiar phrases well before the final token | Success rate at `i>=2` far above chance | `results/processed/paper_numbers.json` → `in_context.generation_olmo2`: 84.0% at `i=2`, 64.8% at `i=3`, 55.6% at `i=4`, 53.6% at `i=5` | Success = the remaining token string appears in the continuation; lenient about trailing text, strict about wording |
| Natural context substantially raises recognition | Same phrases, isolated vs in context | `isolated.generation_olmo2` (59.2 / 31.2 / 20.6 / 17.6 at `i=2..5`) vs `in_context.generation_olmo2` (84.0 / 64.8 / 55.6 / 53.6) | Isolated phrases are a lower bound, not the natural operating condition |
| Recognition is a stable property of the phrase | Per-phrase success distribution vs a binomial null | `per_phrase_consistency_i3` | Computed at `i=3` over the 803 phrases with ≥5 **distinct** contexts; raw instance counts must not be used here (44.8% are duplicates) |
| Patchscopes and real generation measure different things | Matched-instance disagreement | `external.agreement_8L_pooled_i2_5`: 78.7% agreement, 15.7% generation-only, 5.6% patchscopes-only | Primary (8-layer) readout. The 32-layer sweep raises agreement to 81.1% and rebalances the asymmetry to ~3:2, but 7.8% of instances remain patchscopes-only |
| Category ranking depends on the instrument | Same categories, two readouts | `external.category_8L_pooled_i2_5_pct` and `category_pooled_i2_5` | Primary (8L): idiom 57.4 > movie 54.7 > building 47.2, i.e. idioms **best**. Generation: movie 84.5 > building 75.3 > idiom 66.4, i.e. idioms **worst** — a true reversal. Under the 32L sweep idioms fall to the middle, so the reversal weakens to a re-ordering; reported in the same table |
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

### Settled (2026-09-18, by the author)

- **Headline patchscopes budget: eight layers.** The author chose the 8-layer union as the primary readout, matching the August 2026 summary deck. Stated rationale in the paper: it is the layer set on which every per-layer, per-category and probing analysis was run, so those results stay mutually comparable. The 32-layer sweep is reported alongside in Table 2, Table 3 and a dedicated paragraph of §5.4, and is named in Limitations — it costs patchscopes 6–8 points and rebalances the disagreement asymmetry from ~3:1 to ~3:2. The draft does not hide it.
- **The "idioms flip" claim is back in**, because it holds under the 8-layer readout: idioms are the easiest category under patchscopes (57.4%) and the hardest under generation (66.4%). The table's last column shows that under the 32-layer sweep idioms move to the middle rather than the top, so the claim is stated as a reversal that weakens to a re-ordering.

### Settled (2026-09-18, by the agent, at the author's direction)

- **Efficiency framing stays**, as motivation only. Limitations states plainly that no decoding scheme was implemented and no latency was measured. The conclusion names the adaptive multi-token-prediction work as a separate ongoing thread and says explicitly that nothing in this paper depends on its outcome.
- **Idiom provenance resolved.** The set traces to the four sources behind IDIOMEM (Haviv et al., 2023): MAGPIE (511), LIdioms (165), EPIE (81), and EF's public idiom/proverb list (95) — the `EPIC` and `ep_*` labels in `idioms.csv` correspond to EPIE and EF. Verified by matching Haviv et al.'s Table 2 examples and their `Pred.`/`Sim.` filter flags against the CSV's columns. The paper says the set is *not* IDIOMEM (852 vs their 814; we keep items their filters remove).
- **False-positive adjudication stays**, explicitly marked as a single unvalidated LLM judge and as indicative rather than a measured correction. It supports only the directional claim that the reported rates are conservative.
- **Word-aligned cut points: not re-run.** Stays a limitation, with the concrete examples (`istine Chapel`, `ulp Fiction`). Requires cluster access.
- **Probe features: not re-run.** Table 4 and §5.6 are transcribed from the July 2026 run logs into the `external` block of `paper_numbers.json`, which names its source; Limitations says so.
- **Venue: ACL long paper.** Drives the 8-page main body and what goes to appendix.
- **Context duplication corrected.** 44.8% of collected instances repeat a context (collector resumed from an earlier five-context run). This is a bug the deck predates. Aggregate rates are unaffected (64.8% → 64.5% at i=3), but it had inflated the per-phrase bimodality result, which is now computed over distinct contexts against a binomial null and is stronger for it. Documented in §4, Limitations, and `results/README.md`.

### Settled (2026-09-19, from the project assistant's records)

- **The LLM judge was ChatGPT, not Claude**, and the model version was not recorded. The prompt is now reproduced verbatim in Appendix A.8, and Limitations says the run is not exactly reproducible and should be repeated with a named model. An informal second pass by a different assistant is *not* reported, since it used no fixed rubric.
- **Future Lens** (Pal et al., CoNLL 2023) was missing and is the closest prior work — it asks the same question in general form, using the same transplant technique that Patchscopes later generalised. Now cited in the Introduction, Related Work (its own paragraph) and Method, with an explicit statement of what we add: an externally-defined target, a repetition structure across contexts, and a comparison against the model's own behaviour rather than against its own predictions.
- **Confidence is the geometric mean** of per-token probabilities, not the arithmetic mean. Corrected; medians added.
- **A judge-free replication of the false-positive result** was available and is now included: 40.0% of the 412 cases had the model itself confident (>0.7), 10.4% unsure (<0.3). This is the more reproducible of the two analyses and agrees with the adjudication.
- **The L2 dip** under the patchscopes label is now reported in Appendix A.6 as an unresolved artifact (tiny positive class, no multi-seed check run), alongside the L1 surprise (76.6% balanced accuracy after one block).
- **Feature files are intact on the cluster** (1.28 GB and 3.02 GB), so the probe results are reproducible there. Limitations updated to say so rather than implying they are lost.
- **The CLP reproduction** is mentioned in the Conclusion: ~1.34x throughput, not beating a fixed-draft baseline, n=10 so it settles nothing. Framed as bounding the prize rather than as a result of this paper. Summary PDF in `sources/notes/`. CLP itself (Xie & Zhou, arXiv:2606.10935) is now cited, and also appears in Related Work as the closest existing use of the object our probe suggests — a cheap linear read of the hidden state that sets draft length per step.
- **The judge was most likely GPT-5.6**, per the authors, but it was not logged and they are not certain. Stated with that hedge in both Appendix A.8 and Limitations. Still worth rerunning against a recorded model.

### Settled (2026-09-19, by the author)

- **The editorial center is confirmed as the dual claim**, with phenomenon and instrument sharing billing roughly 45/55: *the hidden state mid-phrase carries a usable, phrase-stable signal about what is coming, and how much of it you measure depends on the readout.* The author chose this over two alternatives — leading with the phenomenon and demoting the readout comparison to a methods caveat, or leading with instrument-dependence and treating phrase anticipation as the testbed. No restructuring needed; the draft already reflects this. Contribution 3 stays a claim rather than a caveat.

### Settled (2026-09-19, by cluster rerun)

- **The slide-14 bin chart is dropped**, by the author's decision. Figure~7 (correlation between probe confidence and true phrase difficulty, by layer) already carries the same argument on data that reproduces exactly, so nothing is lost. No further cluster work needed on it.
- **The slide-14 bin-count discrepancy is resolved.** The deck's chart covers 194 phrases; no script or saved file produces that number. Sweeping the minimum-observations filter gives 180 / 190 / 196, so 194 is unreachable and the chart came from an earlier feature snapshot that no longer exists. **180 is correct.** The correlation the paper cites (0.517) was confirmed by direct rerun to four decimal places and belongs to the 180-phrase set — so no number in the paper was affected.
- **One recorded value did not reproduce**: the natural-training MLP at L25 reran as 0.499 against 0.531 recorded. Footnoted in the appendix rather than silently corrected, since the recorded value is what the original analysis produced and the paragraph's claim (the ordering across variants) is unaffected either way.

### Still open for the authors

- Whether to spend cluster time closing the two reproducibility gaps above (word-aligned targets; archived probe features) before submission.

## Interview status

- Last checkpoint: 2026-09-19. The brief was reconstructed by an agent from the August 2026 summary deck, the July 2026 analysis reports, and the raw result files, then corrected against the project assistant's records and two cluster reruns. The editorial center was confirmed by the author on 2026-09-19; it is no longer an inference.
- Everything under **Open decisions** above is settled. The only remaining author judgment is whether to spend cluster time on the two reproducibility gaps before submission.
