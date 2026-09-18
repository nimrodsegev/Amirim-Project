"""
Find disagreement examples between patchscopes and real generation on the
EXACT SAME instances.
"""
import json, torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
BUFFER = 3
LAYERS = [5, 7, 10, 13, 15, 20, 25, 30]
PATCHSCOPE_PROMPT = "Repeat this: X"
REPLACE_TOKEN = "X"
SEED = 29
N_EACH_TARGET = 8

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

patchscope_input_ids = tokenizer.encode(PATCHSCOPE_PROMPT)
patchscope_tensor_base = torch.tensor(patchscope_input_ids).to(model.device)
replace_token_ids = tokenizer.encode(REPLACE_TOKEN, add_special_tokens=False)
replace_mask = patchscope_tensor_base == replace_token_ids[0]
space_replace_ids = tokenizer.encode(' ' + REPLACE_TOKEN, add_special_tokens=False)
if len(space_replace_ids) == 1:
    replace_mask = replace_mask | (patchscope_tensor_base == space_replace_ids[0])
replace_positions = replace_mask.nonzero(as_tuple=False).flatten()
base_inputs_embeds_template = model.get_input_embeddings()(patchscope_tensor_base).unsqueeze(0)


def check_instance(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start_idx = 1 + len(context_ids)
    n_phrase_tokens = len(phrase_ids)
    if n_phrase_tokens < 3:
        return []

    results = []
    with torch.no_grad():
        mi = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        out = model(input_ids=mi, output_hidden_states=True)

        for tok_idx in range(phrase_start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start_idx
            tokens_remaining = n_phrase_tokens - phrase_pos - 1
            if tokens_remaining < 2 or tokens_remaining > 5:
                continue
            rest = tokenizer.decode(phrase_ids[-tokens_remaining:], skip_special_tokens=True)
            max_new = tokens_remaining + BUFFER

            layer_embeddings = torch.stack(
                [out.hidden_states[L][:, tok_idx].squeeze(0) for L in LAYERS], dim=0
            )
            batched_inputs_embeds = base_inputs_embeds_template.repeat(len(LAYERS), 1, 1).clone()
            for batch_idx in range(len(LAYERS)):
                for pos in replace_positions:
                    batched_inputs_embeds[batch_idx, pos] = layer_embeddings[batch_idx]

            ps_success, ps_layer, ps_generated = False, None, None
            ps_layer_texts = {}
            try:
                patchscope_outputs = model.generate(
                    inputs_embeds=batched_inputs_embeds, max_new_tokens=max_new,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                for li, L in enumerate(LAYERS):
                    txt = tokenizer.decode(patchscope_outputs[li], skip_special_tokens=True)
                    ps_layer_texts[L] = txt
                    if not ps_success and rest.lower() in txt.lower():
                        ps_success, ps_layer, ps_generated = True, L, txt
            except Exception:
                pass

            prefix_ids = input_ids[:tok_idx + 1]
            prefix = torch.tensor(prefix_ids).unsqueeze(0).to(model.device)
            g = model.generate(
                input_ids=prefix, max_new_tokens=max_new,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(g[0][len(prefix_ids):], skip_special_tokens=True)
            gen_success = rest.lower() in gen_text.lower()

            results.append({
                "i": tokens_remaining, "target": rest,
                "ps_success": ps_success, "ps_layer": ps_layer, "ps_generated": ps_generated,
                "ps_layer_texts": ps_layer_texts,
                "gen_success": gen_success, "gen_generated": gen_text,
                "context_tail": context_before[-140:],
            })
    return results


print("Loading context data...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

phrases = list(ctx_data.keys())
random.seed(SEED)
random.shuffle(phrases)

ps_only, gen_only = [], []
seen_ps, seen_gen = set(), set()

for idx, phrase in enumerate(phrases):
    if len(ps_only) >= N_EACH_TARGET and len(gen_only) >= N_EACH_TARGET:
        break
    if idx % 20 == 0:
        print(f"  scanned {idx} phrases | ps_only={len(ps_only)} gen_only={len(gen_only)}")
    matches = ctx_data[phrase]
    if not matches:
        continue
    m = matches[0]
    rows = check_instance(m["context_before"], phrase)
    for r in rows:
        if r["ps_success"] and not r["gen_success"] and phrase not in seen_ps and len(ps_only) < N_EACH_TARGET:
            r["phrase"] = phrase; r["category"] = m["category"]
            ps_only.append(r); seen_ps.add(phrase)
        elif r["gen_success"] and not r["ps_success"] and phrase not in seen_gen and len(gen_only) < N_EACH_TARGET:
            r["phrase"] = phrase; r["category"] = m["category"]
            gen_only.append(r); seen_gen.add(phrase)

print("\n" + "=" * 100)
print(f"PATCHSCOPES SUCCEEDED, GENERATION FAILED  ({len(ps_only)} examples)")
print("=" * 100)
for r in ps_only:
    print(f"\nPhrase: {r['phrase']}  (category={r['category']}, i={r['i']})")
    print(f"  context tail: ...{r['context_tail']!r}")
    print(f"  target: {r['target']!r}")
    print(f"  patchscopes (layer {r['ps_layer']}): {r['ps_generated']!r}")
    print(f"  real generation said instead: {r['gen_generated']!r}")

print("\n" + "=" * 100)
print(f"GENERATION SUCCEEDED, PATCHSCOPES FAILED  ({len(gen_only)} examples)")
print("=" * 100)
for r in gen_only:
    print(f"\nPhrase: {r['phrase']}  (category={r['category']}, i={r['i']})")
    print(f"  context tail: ...{r['context_tail']!r}")
    print(f"  target: {r['target']!r}")
    print(f"  real generation: {r['gen_generated']!r}")
    l25_txt = r['ps_layer_texts'].get(25, '(n/a)')
    print(f"  patchscopes L25 said instead: {l25_txt!r}")

print("\nDone. Copy this entire output back to Claude.")
