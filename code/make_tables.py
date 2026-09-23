"""Emit the paper's tables into tables/ from results/processed/paper_numbers.json."""
import json
from pathlib import Path

N = json.load(open("results/processed/paper_numbers.json"))
T = Path("tables"); T.mkdir(exist_ok=True)
IS = ["2", "3", "4", "5"]
LAYERS = [str(l) for l in N["external"]["layers_probed"]]


def write(name, body):
    (T / f"{name}.tex").write_text(body.rstrip() + "\n")
    print("  wrote tables/" + name + ".tex")


# ---- Table 1: dataset ------------------------------------------------------
d = N["dataset"]
rows = []
for cat, label in (("building", "Landmarks"), ("idiom", "Idioms"),
                   ("movie", "Film titles")):
    rows.append(f"{label} & {d['phrases_by_category_source'][cat]:,} & "
                f"{d['phrases_with_context_by_category'][cat]:,} & "
                f"{d['context_instances_by_category'][cat]:,} \\\\")
write("dataset", r"""\begin{table}[t]
\centering\small
\begin{tabular}{@{}lrrr@{}}
\toprule
 & \textbf{Phrases} & \textbf{w/ ctx.} & \textbf{Inst.} \\
\midrule
""" + "\n".join(rows) + rf"""
\midrule
Total & {d['phrases_total']:,} & {d['phrases_with_at_least_one_context']:,} & {d['context_instances']:,} \\
\bottomrule
\end{{tabular}}
\caption{{The phrase set: landmark names (\textit{{the Great Wall of China}}),
idioms (\textit{{the grass is always greener\ldots}}) and film titles
(\textit{{Harry Potter and the Goblet of Fire}}). \textbf{{Phrases}} counts
expressions long enough to admit at least one lookahead position;
\textbf{{w/ ctx.}} those for which a natural occurrence was found in
FineWeb-Edu; \textbf{{Inst.}} the resulting (phrase, context) pairs, of which
{d['phrases_with_10_contexts']:,} phrases contributed the full ten.}}
\label{{tab:dataset}}
\end{{table}}""")


# ---- Table 2: main results -------------------------------------------------
def row(label, path, keys):
    src = N
    for k in path:
        src = src[k]
    cells = []
    for i in keys:
        cells.append(f"{src[i][2]:.1f}" if i in src else "--")
    return f"{label} & " + " & ".join(cells) + r" \\"


e8 = N["external"]["patchscopes_8L_union_in_context_pct"]
write("main_results", r"""\begin{table}[t]
\centering\small
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{4}{c}{\textbf{Lookahead distance} $i$} \\
\cmidrule(lr){2-5}
\textbf{Readout / condition} & 2 & 3 & 4 & 5 \\
\midrule
\multicolumn{5}{l}{\textit{Phrase in isolation}} \\
""" + row("\\quad Patchscopes", ["isolated", "patchscopes_olmo2_32L"], IS) + "\n"
      + row("\\quad Generation", ["isolated", "generation_olmo2"], IS) + r"""
\midrule
\multicolumn{5}{l}{\textit{Phrase in natural context}} \\
""" + "\\quad Patchscopes & " + " & ".join(f"{e8[i]:.1f}" for i in IS) + r" \\" + "\n"
      + row("\\quad Generation", ["in_context", "generation_olmo2"], IS) + r"""
\midrule
\multicolumn{5}{l}{\textit{Robustness: wider Patchscopes layer sweep}} \\
""" + row("\\quad Patchscopes, all 32 layers", ["in_context", "patchscopes_olmo2_32L"], IS) + rf"""
\bottomrule
\end{{tabular}}
\caption{{Recovery rate (\%) of the final $i$ phrase tokens, OLMo-2-7B.
Isolation rows are computed over phrases
({N['isolated']['generation_olmo2']['2'][1]:,} at $i=2$); context rows over
(phrase, context) instances ({N['in_context']['generation_olmo2']['2'][1]:,},
{N['in_context']['generation_olmo2']['3'][1]:,},
{N['in_context']['generation_olmo2']['4'][1]:,} and
{N['in_context']['generation_olmo2']['5'][1]:,} at $i=2\ldots5$).
Patchscopes counts a success if \emph{{any}} probed layer recovers the
continuation, over the eight layers used throughout this paper
(\S\ref{{sec:setup}}); isolation rows sweep all 32. The final row re-runs the
in-context sweep over all 32 layers and is discussed in
\S\ref{{sec:disagreement}}.}}
\label{{tab:main}}
\end{{table}}""")


# ---- Table 3: matched-instance agreement -----------------------------------
a = N["agreement_32L"]
p8 = N["external"]["agreement_8L_pooled_i2_5"]
p32 = N["agreement_32L_pooled_i2_5"]
rows = []
for i in IS:
    r = a[i]
    n = r["n"]
    rows.append(f"{i} & {r['n']:,} & {100*r['both']/n:.1f} & {100*r['patchscopes_only']/n:.1f} & "
                f"{100*r['generation_only']/n:.1f} & {100*r['neither']/n:.1f} & {r['agreement_pct']:.1f} \\\\")
write("agreement", r"""\begin{table}[t]
\centering\small
\begin{tabular}{lrrrrrr}
\toprule
& $n$ & both & \makecell{ps.\\only} & \makecell{gen.\\only} & neither & agree \\
\midrule
\multicolumn{7}{l}{\textit{Eight probed layers, pooled $i=2\ldots5$}} \\
""" + f"\\quad pooled & {p8['n']:,} & {p8['both_pct']:.1f} & {p8['patchscopes_only_pct']:.1f} & "
      f"{p8['generation_only_pct']:.1f} & {p8['neither_pct']:.1f} & {p8['agreement_pct']:.1f} \\\\\n"
      + r"""\midrule
\multicolumn{7}{l}{\textit{All 32 layers, by lookahead distance}} \\
""" + "\n".join("\\quad $i=" + r_[0] + "$" + r_[1:] for r_ in rows) + "\n"
      + f"\\quad pooled & {p32['n']:,} & {100*p32['both']/p32['n']:.1f} & {100*p32['patchscopes_only']/p32['n']:.1f} & "
        f"{100*p32['generation_only']/p32['n']:.1f} & {100*p32['neither']/p32['n']:.1f} & {p32['agreement_pct']:.1f} \\\\\n"
      + r"""\bottomrule
\end{tabular}
\caption{Patchscopes and generation on \emph{identical} (phrase, context, $i$)
instances; all cells are percentages of $n$. Over the eight probed layers the
two readouts agree on 78.7\% of instances and the disagreement is lopsided:
generation recovers what Patchscopes misses roughly three times as often as the
reverse. Widening the Patchscopes sweep to all 32 layers raises agreement to
81.1\% and rebalances the disagreement, but does not remove it: 7.8\% of
instances are still recovered only by Patchscopes.}
\label{tab:agreement}
\end{table}""")


# ---- Table 4: probes -------------------------------------------------------
ps = N["external"]["patchscopes_label_probe"]
ge = N["external"]["generation_label_probe"]
rows = []
for l in LAYERS:
    rows.append(f"{l} & {ps['pos_rate_pct'][l]:.1f} & {ps['balanced_accuracy_pct'][l]:.1f} & "
                f"{ps['precision_pct'][l]:.1f} & {ge['balanced_accuracy_pct'][l]:.1f} & "
                f"{ge['precision_pct'][l]:.1f} \\\\")
write("probes", r"""\begin{table}[t]
\centering\small
\begin{tabular}{rrrrrr}
\toprule
& \multicolumn{3}{c}{\textbf{Patchscopes label}} & \multicolumn{2}{c}{\textbf{Generation label}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
\textbf{Layer} & pos.\ (\%) & bal.\ acc. & prec. & bal.\ acc. & prec. \\
\midrule
""" + "\n".join(rows) + rf"""
\bottomrule
\end{{tabular}}
\caption{{Probes trained on the hidden state at the cut position to predict each
readout's outcome, pooled over $i=2\ldots5$, split by phrase (80/20), training
set class-balanced by downsampling, evaluated on the natural test distribution.
The generation label has a fixed positive rate of {ge['pos_rate_pct']}\% at every
layer; the patchscopes label does not, which is why its raw accuracy is not
comparable across layers. Balanced accuracy for the patchscopes label peaks in
the middle of the model, whereas for the generation label it rises
monotonically with depth.}}
\label{{tab:probes}}
\end{{table}}""")


# ---- Table 5: categories ---------------------------------------------------
c = N["category_pooled_i2_5"]
c8 = N["external"]["category_8L_pooled_i2_5_pct"]
rows = []
for cat, label in (("building", "Landmarks"), ("idiom", "Idioms"), ("movie", "Film titles")):
    rows.append(f"{label} & {c['generation'][cat]['n']:,} & {c8[cat]:.1f} & "
                f"{c['generation'][cat]['success_pct']:.1f} & "
                f"{c['patchscopes_32L'][cat]['success_pct']:.1f} \\\\")
write("categories", r"""\begin{table}[t]
\centering\small
\begin{tabular}{lrrrr}
\toprule
& & \multicolumn{2}{c}{\textbf{Recovery (\%)}} & \\
\cmidrule(lr){3-4}
\textbf{Category} & \textbf{Inst.} & patchscopes & generation & \small{(32L)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Recovery by phrase category, pooled over $i=2\ldots5$, in natural
context. The two readouts reverse the ordering of idioms: under Patchscopes
idioms are the \emph{easiest} category, under generation the hardest. The final
column repeats the Patchscopes measurement over all 32 layers, under which
idioms fall to the middle of the ranking; the reversal against generation is
therefore weaker but does not disappear.}
\label{tab:categories}
\end{table}""")


print("done")


# ---- Table 6: replication across models ------------------------------------
iso, ctxn = N["isolated"], N["in_context"]
q = N["external"]["qwen25_14b_generation_pct"]


def cells(src, keys, fmt="{:.1f}"):
    return " & ".join(fmt.format(src[k][2]) if k in src else "--" for k in keys)


def qcells(src, keys):
    return " & ".join(f"{src[k]:.1f}" if k in src else "--" for k in keys)


write("replication", r"""\begin{table}[t]
\centering\small
\begin{tabular}{llrrrr}
\toprule
& & \multicolumn{4}{c}{$i$} \\
\cmidrule(lr){3-6}
\textbf{Model} & \textbf{Readout} & 2 & 3 & 4 & 5 \\
\midrule
\multicolumn{6}{l}{\textit{Phrase in isolation}} \\
""" + f"OLMo-2-7B & patchscopes & {cells(iso['patchscopes_olmo2_32L'], IS)} \\\\\n"
      + f"OLMo-3-7B & patchscopes & {cells(iso['patchscopes_olmo3_32L'], IS)} \\\\\n"
      + f"OLMo-2-7B & generation & {cells(iso['generation_olmo2'], IS)} \\\\\n"
      + f"Qwen2.5-14B & generation & {qcells(q['isolated'], IS)} \\\\\n" + r"""\midrule
\multicolumn{6}{l}{\textit{Phrase in natural context}} \\
""" + f"OLMo-2-7B & patchscopes & {cells(ctxn['patchscopes_olmo2_32L'], IS)} \\\\\n"
      + f"OLMo-3-7B & patchscopes & {cells(ctxn['patchscopes_olmo3_32L'], IS)} \\\\\n"
      + f"OLMo-2-7B & generation & {cells(ctxn['generation_olmo2'], IS)} \\\\\n"
      + f"Qwen2.5-14B & generation & {qcells(q['in_context'], IS)} \\\\\n" + r"""\bottomrule
\end{tabular}
\caption{Recovery rate (\%) across models, on the same phrases, contexts and cut
points. Patchscopes rows are the union over all 32 layers. Qwen2.5-14B, at
roughly twice the parameters of OLMo-2-7B, is slightly behind it in isolation
and slightly ahead of it in context.}
\label{tab:replication}
\end{table}""")
