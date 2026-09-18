"""
Collect up to 10 natural-context occurrences of each phrase from FineWeb-Edu.
Resumes from existing data/phrases_with_context.json if present (old cap was 5,
new matches will top up existing phrases to 10).

For proper-noun phrases (mostly capitalized, e.g. movies and buildings), only
keep matches where the phrase appears in the document with at least 70% of the
original capitalization preserved. This avoids false positives like "before
sunrise" about turkey behavior being counted as the movie Before Sunrise.
"""
import csv, json, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
MATCHES_PER_PHRASE = 10
CONTEXT_TOKENS = 150
DOC_CAP = 10_000_000
OUT_PATH = Path("data/phrases_with_context_v2.json")
SEED_PATH = Path("data/phrases_with_context.json")  # existing data to seed from
PROGRESS_EVERY = 25_000

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading phrases...")
phrases = []
for path, cat in [("data/buildings_clean.csv", "building"),
                  ("data/idioms.csv", "idiom"),
                  ("data/imdb_top250_multiword.csv", "movie")]:
    with open(path) as f:
        reader = csv.DictReader(f)
        col = reader.fieldnames[0]
        for row in reader:
            p = row[col].strip()
            if p: phrases.append((p, cat))
print(f"  {len(phrases)} phrases total")

def is_proper_noun(phrase):
    words = [w for w in phrase.split() if any(c.isalpha() for c in w)]
    if not words: return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) >= 0.5

def passes_case_filter(phrase, phrase_as_found):
    if not is_proper_noun(phrase): return True  # idioms: no check
    orig_caps = [w[0].isupper() for w in phrase.split() if w and w[0].isalpha()]
    found_caps = [w[0].isupper() for w in phrase_as_found.split() if w and w[0].isalpha()]
    if len(orig_caps) == 0 or len(found_caps) != len(orig_caps): return False
    matches = sum(1 for o, f in zip(orig_caps, found_caps) if o == f)
    return matches / len(orig_caps) >= 0.7

phrases_lc = [(p.lower(), p, cat) for p, cat in phrases]

# Seed from existing file if present
matches = {p: [] for p, _ in phrases}
if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        saved = json.load(f)
    for p, _ in phrases:
        if p in saved: matches[p] = saved[p]
    print(f"  Resumed from {OUT_PATH}")
elif SEED_PATH.exists():
    with open(SEED_PATH) as f:
        saved = json.load(f)
    for p, _ in phrases:
        if p in saved:
            # keep only matches that pass the case filter
            kept = [m for m in saved[p] if passes_case_filter(p, m["phrase_as_found"])]
            matches[p] = kept
    print(f"  Seeded from {SEED_PATH} (applied case filter)")

completed = {p for p, ms in matches.items() if len(ms) >= MATCHES_PER_PHRASE}
total_matches = sum(len(m) for m in matches.values())
print(f"  Already complete: {len(completed)}/{len(phrases)}")
print(f"  Existing matches: {total_matches}")

def save():
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

def extract_context(text, phrase, phrase_lc):
    text_lc = text.lower()
    idx = text_lc.find(phrase_lc)
    if idx == -1: return None
    before_text = text[:idx]
    phrase_actual = text[idx:idx + len(phrase)]
    if not passes_case_filter(phrase, phrase_actual):
        return None
    before_ids = tok.encode(before_text, add_special_tokens=False)
    if len(before_ids) > CONTEXT_TOKENS:
        before_ids = before_ids[-CONTEXT_TOKENS:]
    before_snippet = tok.decode(before_ids, skip_special_tokens=True)
    return {
        "context_before": before_snippet.strip(),
        "phrase_as_found": phrase_actual,
        "n_context_tokens": len(before_ids),
    }

print("\nStreaming FineWeb-Edu sample-10BT...")
stream = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

start = time.time()
docs_seen = 0
last_save = total_matches

try:
    for sample in stream:
        docs_seen += 1
        if docs_seen >= DOC_CAP:
            print(f"\nHit doc cap. Stopping.")
            break
        if len(completed) == len(phrases):
            print(f"\nAll phrases complete. Stopping.")
            break

        text = sample.get("text", "")
        if not text: continue
        text_lc = text.lower()

        for phrase_lc, phrase, cat in phrases_lc:
            if phrase in completed: continue
            if phrase_lc not in text_lc: continue
            if len(matches[phrase]) >= MATCHES_PER_PHRASE:
                completed.add(phrase); continue
            ctx = extract_context(text, phrase, phrase_lc)
            if ctx is None: continue
            ctx["category"] = cat
            ctx["source_doc_id"] = docs_seen
            matches[phrase].append(ctx)
            total_matches += 1
            if len(matches[phrase]) >= MATCHES_PER_PHRASE:
                completed.add(phrase)

        if docs_seen % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            print(f"  {docs_seen:>10,} docs | {len(completed):>4}/{len(phrases)} complete | "
                  f"{total_matches} matches | {elapsed:.0f}s")
            if total_matches - last_save >= 300:
                save(); last_save = total_matches

except KeyboardInterrupt:
    print("\nInterrupted. Saving...")

save()
elapsed = time.time() - start
print(f"\n{'='*70}")
print(f"Streamed {docs_seen:,} docs in {elapsed:.0f}s")
print(f"Phrases with 10 matches: {len(completed)}/{len(phrases)}")
print(f"Total matches: {total_matches}")

cat_stats = {"building": {"p": 0, "m": 0}, "idiom": {"p": 0, "m": 0}, "movie": {"p": 0, "m": 0}}
distribution = {i: 0 for i in range(11)}
for phrase, cat in phrases:
    ms = matches[phrase]
    cat_stats[cat]["p"] += 1
    cat_stats[cat]["m"] += len(ms)
    distribution[min(len(ms), 10)] += 1

print(f"\nBy category:")
for c, s in cat_stats.items():
    avg = s["m"] / s["p"] if s["p"] else 0
    print(f"  {c:<10}: {s['m']:>5} matches across {s['p']} phrases (avg {avg:.2f})")

print(f"\nDistribution of match counts:")
for k in range(11):
    print(f"  {k:>2} matches: {distribution[k]:>4} phrases")
