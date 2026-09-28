#!/usr/bin/env bash
# =============================================================================
# Deduplicated rerun, for the compute cluster.
#
# Why: data/phrases_with_context_v2.json holds 9,955 (phrase, context) rows, of
# which 44.8% are exact repeats of a context already present for the same
# phrase. Decoding is greedy, so a repeat re-scores the same trial and inflates
# the nominal sample without adding information.
#
# How: every experiment script reads that one file, so we deduplicate the file
# itself and leave the scripts untouched. The original is kept alongside, and
# stage 0 is reversible (see RESTORE at the bottom).
#
# Run from the directory that holds data/ and results/:
#     bash rerun_dedup.sh              # stages 0-2 (OLMo-2, needed for the paper)
#     bash rerun_dedup.sh --with-replications   # also OLMo-3 and Qwen2.5-14B
#
# Send back: rerun_dedup_output.tar.gz (created at the end).
# =============================================================================
set -euo pipefail

WITH_REPL=0
[[ "${1:-}" == "--with-replications" ]] && WITH_REPL=1

# Where the experiment scripts live. Override with: SCRIPTS=/path bash rerun_dedup.sh
SCRIPTS="${SCRIPTS:-}"
if [[ -z "$SCRIPTS" ]]; then
  for c in "code/phrase_recognition" "phrase_recognition" "."; do
    [[ -f "$c/rerun_with_context_full.py" ]] && { SCRIPTS="$c"; break; }
  done
fi
[[ -n "$SCRIPTS" ]] || { echo "ERROR: cannot find rerun_with_context_full.py. Set SCRIPTS=/path/to/scripts"; exit 1; }

DATA="data/phrases_with_context_v2.json"
[[ -f "$DATA" ]] || { echo "ERROR: $DATA not found. Run from the directory containing data/."; exit 1; }

mkdir -p results logs rerun_dedup_output
LOG="logs/rerun_dedup_$(date +%Y%m%d_%H%M%S).log"
echo "scripts: $SCRIPTS"
echo "log:     $LOG"
echo

# Run a step, tee its output to the log, keep a per-step copy, never die silently.
run () {
  local name="$1"; shift
  echo "=============================================================" | tee -a "$LOG"
  echo ">>> $name   [$(date +%H:%M:%S)]"                                | tee -a "$LOG"
  echo "=============================================================" | tee -a "$LOG"
  if [[ ! -f "$SCRIPTS/$name" ]]; then
    echo "!!! SKIPPED: $SCRIPTS/$name not found" | tee -a "$LOG"; return 0
  fi
  if python3 "$SCRIPTS/$name" 2>&1 | tee -a "$LOG" | tee "rerun_dedup_output/${name%.py}.out"; then
    echo "<<< $name OK" | tee -a "$LOG"
  else
    echo "!!! $name FAILED (continuing so the rest still runs)" | tee -a "$LOG"
  fi
  echo | tee -a "$LOG"
}

# ---------------------------------------------------------------- stage 0 ----
# Deduplicate on (phrase, context_before), keeping the first occurrence.
# Idempotent: if the backup already exists we assume stage 0 has run.
echo ">>> stage 0: deduplicate $DATA" | tee -a "$LOG"
if [[ -f "data/phrases_with_context_v2_ORIGINAL.json" ]]; then
  echo "    backup already present, skipping dedup" | tee -a "$LOG"
else
  cp "$DATA" "data/phrases_with_context_v2_ORIGINAL.json"
  # back up the result files the rerun will overwrite
  for f in results/*.json; do [[ -e "$f" ]] && cp "$f" "${f%.json}_PREDEDUP.json"; done
  python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, collections
SRC = "data/phrases_with_context_v2_ORIGINAL.json"
DST = "data/phrases_with_context_v2.json"
data = json.load(open(SRC))
out, kept, dropped = {}, 0, 0
for phrase, matches in data.items():
    seen, keep = set(), []
    for m in matches:
        key = m["context_before"]
        if key in seen:
            dropped += 1
            continue
        seen.add(key); keep.append(m)
    if keep:
        out[phrase] = keep; kept += len(keep)
json.dump(out, open(DST, "w"), indent=2)
before = sum(len(v) for v in data.values())
print(f"    phrases:   {len(data)} -> {len(out)}")
print(f"    instances: {before} -> {kept}   ({dropped} exact repeats removed, "
      f"{100*dropped/before:.1f}%)")
# how many phrases still have enough contexts for the per-phrase analysis
n5 = sum(1 for v in out.values() if len(v) >= 5)
print(f"    phrases with >=5 distinct contexts: {n5}")
PY
fi
echo | tee -a "$LOG"

# ---------------------------------------------------------------- stage 1 ----
# GPU. Everything that reads the context file and runs the model.
# rerun_with_context_full is the long one; the rest are shorter.
run rerun_with_context_full.py            # -> results/lookahead_with_context_full.json
run generate_truth_context.py             # -> results/generation_truth_context.json
run extract_probing_features.py           # -> data/probing_features.npz
run extract_generation_probing_features.py # -> data/generation_probing_features.npz
run extract_and_probe_early_layers.py     # -> data/probing_features_early.npz
run task_2_4_confidence.py                # -> data/generation_confidence.json

# ---------------------------------------------------------------- stage 2 ----
# CPU. Reads the .npz files stage 1 just wrote.
run train_probes_pooled.py                # -> results/probing_pooled.json
run correlation_per_i.py                  # -> results/correlation_per_i.json
run category_breakdown.py                 # generation rates by category
run category_breakdown_patchscopes.py     # patchscopes rates by category
run get_precision.py                      # probe precision
run task_4_false_positives.py             # false-positive counts

# ---------------------------------------------------------------- stage 3 ----
# Optional: the replication table. Two more model loads, Qwen2.5-14B is large.
if [[ $WITH_REPL -eq 1 ]]; then
  run rerun_olmo3_with_context.py
  run test_qwen_generation_truth.py
else
  echo ">>> stage 3 (replications) skipped. Re-run with --with-replications to include." | tee -a "$LOG"
fi

# ---------------------------------------------------------------- collect ----
echo "=============================================================" | tee -a "$LOG"
echo ">>> collecting results" | tee -a "$LOG"
for f in results/lookahead_with_context_full.json \
         results/generation_truth_context.json \
         results/correlation_per_i.json \
         results/probing_pooled.json \
         results/lookahead_olmo3_with_context.json ; do
  [[ -f "$f" ]] && cp "$f" rerun_dedup_output/ && echo "    + $f" | tee -a "$LOG"
done
cp "$LOG" rerun_dedup_output/
tar czf rerun_dedup_output.tar.gz rerun_dedup_output
echo
echo "DONE. Send back:  $(pwd)/rerun_dedup_output.tar.gz"
echo "      size: $(du -h rerun_dedup_output.tar.gz | cut -f1)"
echo
echo "To undo stage 0:"
echo "  mv data/phrases_with_context_v2_ORIGINAL.json data/phrases_with_context_v2.json"
echo "  for f in results/*_PREDEDUP.json; do mv \"\$f\" \"\${f%_PREDEDUP.json}.json\"; done"
