# Legacy result files

These files are not read by `code/make_paper_numbers.py` and no number in the
paper comes from them. They are kept as a record of analyses that were run
during the project and are not part of the final pipeline.

- `probing_pooled.json`, `probing_all_layers.json`, `probe_multi_layer.json`,
  `probe_mlp_regularized.json`: probe sweeps across classifiers, layer sets and
  regularisation settings.
- `probing_bins_seeds.json`: the per-phrase bin chart and multi-seed check
  behind a slide that is not in the paper.

`probing_pooled.json` was regenerated in the September 2026 deduplicated rerun;
the other four predate deduplication and their numbers do not match the paper.
