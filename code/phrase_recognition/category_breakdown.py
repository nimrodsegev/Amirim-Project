"""
Category breakdown for the real-generation, with-context results.
Uses the already-saved data/generation_probing_metadata.json and
data/generation_probing_features.npz, no model run needed.
"""
import json
import numpy as np

data = np.load("data/generation_probing_features.npz")
success = data["success"]

with open("data/generation_probing_metadata.json") as f:
    meta = json.load(f)
categories = np.array([r["category"] for r in meta["rows"]])

total_n = len(success)
total_success = success.sum()

print(f"Total instances (pooled i=2-5): {total_n}")
print(f"Total successes: {total_success} ({100*total_success/total_n:.1f}%)\n")

print(f"{'category':<12}{'n':<8}{'success%':<11}{'share_of_all':<14}{'share_of_successes'}")
print("-" * 65)
for cat in sorted(set(categories)):
    mask = categories == cat
    n_cat = mask.sum()
    succ_cat = success[mask].sum()
    succ_rate = 100 * succ_cat / n_cat
    share_of_all = 100 * n_cat / total_n
    share_of_successes = 100 * succ_cat / total_success
    print(f"{cat:<12}{n_cat:<8}{succ_rate:<11.1f}{share_of_all:<14.1f}{share_of_successes:.1f}")

print("\nDone. Copy this entire output back to Claude.")
