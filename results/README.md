# Results

`raw/` holds the per-trial analysis outputs from the cluster runs. `processed/`
holds `paper_numbers.json`, the single source for every value the paper reports.

## How the paper's numbers are produced

```bash
python3 code/make_paper_numbers.py   # raw/ -> processed/paper_numbers.json
python3 code/make_tables.py          # processed/ -> tables/*.tex
python3 code/make_figures.py         # processed/ + raw/ -> figures/*.pdf
python3 code/make_schematic.py       # -> figures/protocol.pdf, figures/motivation.pdf
```

Every number in the main text except Table 7 and the false-positive adjudication
in §5.5 is recomputed by these scripts. Do not edit `tables/*.tex` or
`figures/*.pdf` by hand; change the script and re-run.

## What is in `raw/`

| File | Model | Condition | Measurement |
|---|---|---|---|
| `lookahead_analysis_fixed.json` | OLMo-2-7B | isolated | Patchscopes, 32 layers |
| `lookahead_5x_fixed.json` | OLMo-2-7B | isolated | Patchscopes, 5-slot prompt ablation |
| `lookahead_olmo3_fixed.json` | OLMo-3-7B | isolated | Patchscopes, 32 layers |
| `generation_truth_naked.json` | OLMo-2-7B | isolated | greedy generation |
| `lookahead_with_context_full.json` | OLMo-2-7B | FineWeb-Edu context | Patchscopes, 32 layers |
| `lookahead_olmo3_with_context.json` | OLMo-3-7B | FineWeb-Edu context | Patchscopes, 32 layers |
| `generation_truth_context.json` | OLMo-2-7B | FineWeb-Edu context | greedy generation |
| `correlation_per_i.json` | OLMo-2-7B | per-phrase correlation, 8 layers |

`lookahead_with_context_full.json` and `generation_truth_context.json` are
row-aligned: record *k* of one is the same (phrase, context) instance as record
*k* of the other. `make_paper_numbers.py` asserts this before computing the
agreement table.

Outputs the current pipeline does not read are in `legacy/`.

## What to know when reading these files

**The contexts are deduplicated.** Each (phrase, context) pair appears once:
5,498 pairs over 1,096 phrases, 13,613 trials at *i* = 2 to 5.

**The layer budget differs by analysis.** Patchscopes unions are over all 32
layers in the `lookahead_*` files, and over 8 layers (5, 7, 10, 13, 15, 20, 25,
30) wherever per-trial hidden states had to be stored. The choice moves the
Patchscopes rate by 6 to 8 points, so any rate should say which budget produced
it.

**Two results are not regenerated here.** The probing experiments need the
per-trial hidden states, `probing_features.npz` (1.28 GB) and
`generation_probing_features.npz` (3.02 GB), which stay on the cluster. Table 7
and the false-positive adjudication are transcribed from the run logs into the
`external` block of `processed/paper_numbers.json`, which names their source.

**The adjudication used a single automatic judge.** ChatGPT (GPT-5.6 Sol) over
the 212 false positives of the L25 generation-label probe. The prompt is in the
paper's appendix so the run can be repeated, but one automatic judge is not a
substitute for human annotation. The split by the model's own confidence, which
uses no judge, points the same way.
