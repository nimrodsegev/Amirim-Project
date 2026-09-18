# Results

Keep immutable analysis outputs in `raw/` and derived, paper-ready data in `processed/`. Record provenance for every reported value.

## How the paper's numbers are produced

```bash
python3 code/make_paper_numbers.py   # raw/ -> processed/paper_numbers.json
python3 code/make_tables.py          # processed/ -> tables/*.tex
python3 code/make_figures.py         # processed/ + raw/ -> figures/*.pdf
```

Every number in the main text except Table 4 (probes) and the false-positive
adjudication in §5.6 is recomputed by these scripts. Do not edit `tables/*.tex`
or `figures/*.pdf` by hand; change the script and re-run.

## What is in `raw/`

| File | Model | Condition | Readout |
|---|---|---|---|
| `lookahead_analysis_fixed.json` | OLMo-2-7B | isolated | patchscopes, 32 layers |
| `lookahead_5x_fixed.json` | OLMo-2-7B | isolated | patchscopes, 5-slot prompt ablation |
| `lookahead_olmo3_fixed.json` | OLMo-3-7B | isolated | patchscopes, 32 layers |
| `generation_truth_naked.json` | OLMo-2-7B | isolated | greedy generation |
| `lookahead_with_context_full.json` | OLMo-2-7B | FineWeb-Edu context | patchscopes, 32 layers |
| `lookahead_olmo3_with_context.json` | OLMo-3-7B | FineWeb-Edu context | patchscopes, 32 layers |
| `generation_truth_context.json` | OLMo-2-7B | FineWeb-Edu context | greedy generation |
| `correlation_per_i.json`, `probing_*.json`, `probe_*.json` | OLMo-2-7B | probe metrics from the 8-layer runs |

`lookahead_with_context_full.json` and `generation_truth_context.json` are
row-aligned: record *k* of one is the same (phrase, context) instance as record
*k* of the other. `make_paper_numbers.py` asserts this before computing the
agreement table.

## Known issues with the raw data

- **Duplicated contexts.** 44.8% of the 9,955 instances repeat a context already
  present for the same phrase, because the collector resumed from an earlier
  five-context run. There are 5,498 distinct (phrase, context) pairs. Aggregate
  rates are essentially unaffected (64.8% vs 64.5% at *i*=3), but per-phrase
  counts must be deduplicated and the effective sample size is about half the
  instance count. See `context_duplication` in `processed/paper_numbers.json`.
- **Layer budget differs by analysis.** Patchscopes unions are over all 32
  layers in the `lookahead_*` files and over 8 layers (5, 7, 10, 13, 15, 20, 25,
  30) wherever hidden states had to be stored. The choice moves the reported
  patchscopes rate by 6–8 points; always state which was used.
- **Not everything is reproducible here.** The probing experiments need
  per-trial hidden states — `probing_features.npz` (1.28 GB) and
  `generation_probing_features.npz` (3.02 GB) — which are intact on the compute
  cluster but too large to distribute. Table 4 and the false-positive
  adjudication are transcribed from the July 2026 run logs into the `external`
  block of `processed/paper_numbers.json`, which names their source.
- **The false-positive adjudication is not exactly reproducible.** It was run
  through ChatGPT (most likely GPT-5.6; not logged). The prompt is in Appendix
  A.8 of the paper. The judge-free split by model confidence, reported
  alongside it, is reproducible and points the same way.
- **One deck figure is deliberately absent.** The slide-14 chart (L25
  per-phrase prediction bins, `i=3` only, four probe variants) cannot be
  regenerated: `probe_bins_and_seeds.py` computed the per-phrase
  `(avg_pred, actual_rate)` points in memory and saved only the correlations.
  It is reproducible by dumping those rows and rerunning against
  `probing_features.npz`. Note before doing so: the deck's bin counts sum to
  194 phrases, while `probing_bins_seeds.json` records 180 for `i=3_only` at
  L25 and 203 for `all_i` — so the chart and the saved correlation come from
  runs with different phrase counts. Resolve that before putting the figure in
  the paper.
