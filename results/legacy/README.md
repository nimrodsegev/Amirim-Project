# Legacy result files

`code/make_paper_numbers.py` does not read these files and no number in the
paper comes from them. They record probe analyses that are not part of the final
pipeline: sweeps across classifiers, layer sets and regularisation settings
(`probing_pooled.json`, `probing_all_layers.json`, `probe_multi_layer.json`,
`probe_mlp_regularized.json`), and a per-phrase bin chart with a multi-seed
check (`probing_bins_seeds.json`).

All but `probing_pooled.json` were produced before the contexts were
deduplicated, so their numbers do not match the paper.
