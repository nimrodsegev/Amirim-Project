# Citation practice for NLP/LLM papers

A citation should let the reader identify who is responsible for a claim, inspect its evidence, and understand how prior work relates to the current argument. It is not decoration and does not by itself repair an unsupported sentence.

## Rhetorical function

Before inserting a citation, identify its job:

- **Attribution:** credit the origin of a method, dataset, result, term, or argument.
- **Evidence:** support a factual or field-level claim with work that actually establishes it.
- **Positioning:** show agreement, extension, contrast, or a remaining limitation.
- **Navigation:** direct readers to details that are useful but not reproduced here.

Write the relationship in prose. A cluster such as `\citep{a,b,c}` does not tell readers whether the papers agree, use the same method, or merely touch the topic.

## Integrate citations into sentences

Use a textual citation when the cited authors or their action is part of the sentence:

```tex
\citet{yu-ettinger-2020-assessing} find that ...
```

Use a parenthetical citation when the proposition is the grammatical focus:

```tex
This pattern also appears in phrase-level probing \citep{yu-ettinger-2020-assessing}.
```

Do not write `\citet{key} shows` when the cited paper has plural authors and the rendered grammar would be wrong; phrase the sentence so the generated author string agrees with the verb. Avoid generic `\cite{...}` because its rendering varies across styles.

Place a citation immediately after the smallest claim it supports. Do not leave a paragraph-final citation ambiguously responsible for several preceding claims. When one sentence makes multiple independently sourced claims, split the sentence or attach citations locally.

## Match citation scope to claim scope

- Verify that every cited paper supports the full proposition, not just a nearby topic.
- Do not cite a paper for a claim it only repeats from another source; cite the primary work when practical.
- Cite the authoritative published version rather than an earlier arXiv version when one exists, unless the version difference matters.
- Do not cite a paper you have not read closely enough to verify the attributed claim.
- Distinguish a paper's demonstrated result from its speculation, motivation, or related-work summary.
- Use page or section pointers in source notes for claims likely to be contested, even when the final citation style does not display them.

For field-level claims such as “prior work generally assumes X,” one example rarely establishes prevalence. Either support the scope with representative evidence, narrow the wording, or state that the selected papers are examples.

## Synthesize rather than inventory

Group papers only when they support the same useful generalization. Give unequal space when their relevance differs. A related-work paragraph should normally establish an axis, synthesize what a body of work achieves, and end with the present paper's precise relation to it.

Avoid:

- one sentence per paper with no conclusion;
- citations added only because a paper shares keywords;
- “Several works ...” followed by a list whose members do materially different things;
- citation density that obscures the paragraph's own claim.

## Technical workflow

1. Search the project's literature audit or parsed source notes first. Reuse sources already read and scoped for the claim when they support it.
2. If the audit has no adequate source, search for primary work and read the relevant passage before citing it. Do not expand a citation cluster merely because more related papers exist.
3. Search the generated ACL Anthology shards for the verified paper and use its canonical bibkey.
4. Put verified entries absent from the Anthology in `custom.bib`.
5. Prefer `\citep` for parenthetical and `\citet` for textual citations. Use `\citealp`, `\citeyearpar`, or `\citeposs` only when their specific rendering is needed.
6. Check title, authors, year, venue, DOI/URL, and whether a later published version exists. Protect acronyms and case-sensitive names in BibTeX titles with braces when needed.
7. Avoid duplicate keys across `custom.bib` and the Anthology shards.
8. Compile through BibTeX and inspect warnings. Run the bundled citation checker before sharing.

Run from the paper root:

```bash
python3 .agents/skills/nlp-paper-writing/scripts/check_citations.py .
```

The checker finds undefined citation keys, duplicate BibTeX keys, missing bibliography declarations/files, generic `\cite`, and uncited entries in `custom.bib`. It cannot determine whether a source genuinely supports the prose; that requires reading and judgment.

## Revision rules

Treat a citation as attached to a specific claim. Do not move, remove, or broaden the surrounding claim during stylistic revision without rechecking the source relationship. Never invent a key or insert a plausible citation from memory. Use an explicit placeholder such as `\placeholder{citation needed}` when verification remains unresolved.
