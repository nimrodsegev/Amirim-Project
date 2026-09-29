# Paper brief

A claim-and-evidence map for the paper. Written by `code/make_paper_brief.py`
from `results/processed/paper_numbers.json`, so the values below come from the
same source as the paper's.

## Identity

- Title: Do Language Models Recognize a Phrase Before Reading It?
- Amirim final project (HUJI, CS 68101), written in ACL format so it can be
  extended toward a future ACL-style submission. No venue or deadline is fixed,
  so no page limit binds.
- Type: empirical study with a small methodological contribution.
- Intended reader: an NLP reader who knows how language models work but nothing
  about this project.

## One sentence

When a language model is partway through a familiar multi-word expression, does
its hidden state already carry enough information to recover the rest?

## Data

- 1,164 fixed expressions: 852 idioms,
  127 landmarks, 185 film titles.
- 5,498 distinct (phrase, context) pairs from FineWeb-Edu, covering
  1,096 phrases. Repeated contexts are collapsed; only 12
  phrases reach the full ten.
- 13,613 trials at `i` in 2..5.

## Claims and evidence

| Claim | Evidence | Where | Scope or caveat |
|---|---|---|---|
| Familiar phrases are recovered several tokens early | Patchscopes, in context, `i`=2/3/4/5 | `external.patchscopes_8L_union_in_context_pct`: 71.9 / 56.6 / 42.3 / 37.3 | Eight probed layers, the budget used throughout the paper |
| The model's own continuation recovers more | Generation, in context | `in_context.generation_olmo2`: 83.3 / 64.5 / 55.3 / 53.0 | Generation is the baseline, not a second reading of the state |
| Natural context raises recovery a lot | Same phrases, isolated vs in context | generation 59.2 / 31.2 / 20.6 / 17.6 isolated; Patchscopes 32L 57.8 / 30.9 / 19.4 / 16.5 isolated vs 78.4 / 63.2 / 50.1 / 45.2 in context | Isolation is not the natural operating condition |
| Outcomes cluster strongly by phrase | Per-phrase success vs a binomial null | `per_phrase_consistency_i3`: 31.5% of phrases recovered in >=90% of contexts against a null of 10.1% | At `i`=3 over the 803 phrases with >=5 distinct contexts. Shows clustering, **not** that phrase identity causes it: frequency, length, tokenization, category and context similarity are not separated |
| A probe on one hidden state predicts completion | MLP on a single hidden state, generation label | `external.generation_label_probe`: 74.1% balanced accuracy, 84.6% precision at L30 | One phrase-level split at seed 42; features are standardised before training. Not recomputable from this repo, the feature arrays stay on the cluster |
| The two measurements mostly agree | Matched instances | `external.agreement_8L_pooled_i2_5`: 77.8% agreement, 16.6% generation-only, 5.6% Patchscopes-only | Widening to 32 layers lifts agreement to 80.4% |
| The layer budget moves Patchscopes rates | 8 layers vs all 32 | 71.9 / 56.6 / 42.3 / 37.3 against 78.4 / 63.2 / 50.1 / 45.2 | Six to eight points. Always state which budget a Patchscopes rate used |
| The two rank categories differently | Same categories, both measurements | `external.category_8L_pooled_i2_5_pct` and `category_pooled_i2_5` | Patchscopes 8L: movie 59.9 > idiom 58.0 > building 46.4. Generation: movie 83.9 > building 74.1 > idiom 65.7. Film titles lead under both; landmarks and idioms swap. The 32L sweep gives the same ordering as 8L (movie 67.4 > idiom 64.7 > building 55.7) |
| Exact matching understates real knowledge | Judge over the probe's false positives | `external.false_positive_llm_judgement`: 59.9% valid alternate, 22.2% partially right, 17.9% unrelated, n=212 | One automatic judge (ChatGPT, GPT-5.6 Sol), no human validation. Probe-selected, not a sample of all failures, so it does not correct the headline rates. The judge-free confidence split agrees: 46.2% of those cases had model confidence above 0.7 |
| Probe confidence tracks phrase identity | Correlation with true per-phrase rate | `probe_phrase_correlation.by_layer_i3`: peaks at r=0.611 (L10), 0.441 at L5, 0.527 at L30 | Patchscopes label only; the equivalent on generation-label features was not run |

## Terminology

| Concept | Use | Avoid |
|---|---|---|
| Tokens still unread at the cut | **lookahead distance `i`** | "position", "offset" |
| Reading a hidden state via an injected prompt | **Patchscopes** | "readout"; "the model knows" without qualification |
| Greedy continuation from the true prefix | **generation**, our **baseline** | "readout"; "ground truth" |
| Holding information about tokens not yet read | **looking ahead / lookahead** | "anticipation" |
| A phrase recovered at some layer | **recovered / recognized at distance `i`** | "predicted" |
| Fixed multi-word expression | **phrase** | "entity", "collocation" |

## How to read the numbers

- The primary Patchscopes budget is **eight layers**, the set on which the
  per-layer, per-category and probing analyses were run. The 32-layer sweep is
  reported alongside wherever it exists, and the two differ by six to eight
  points.
- Each (phrase, context) pair appears once; the contexts are deduplicated.
- The category comparison covers `i` in 2..5. Over all distances the ordering
  differs, because the long-lookahead rows carry different category proportions.
- The adjudication used one automatic judge, ChatGPT (GPT-5.6 Sol), with the
  prompt reproduced in the appendix.

## What the paper does not establish

- No confidence intervals. The right resampling unit is the phrase, so a
  phrase-clustered bootstrap over the central comparisons is the remaining
  statistical work.
- No frequency-matched control, and pretraining frequency is assumed rather than
  counted. Those two analyses would do most to separate phrase identity from
  exposure.
- No efficiency claim: no decoding scheme, no latency measurement.
- The in-context replication on OLMo-3 and Qwen, the generation-label probe at
  the earliest layers, held-out-category generalisation, layer-group pooling and
  the confidence breakdown by probe outcome are not reported.
