---
name: nlp-paper-writing
description: Interview authors, draft, revise, and review LaTeX papers about NLP and LLMs using a paper brief, related-paper sources, claim-evidence discipline, citation practice, editorial selection, section-aware guidance, and visible editing macros. Use for eliciting a paper story, paper outlining, section drafting, prose revision, author-comment resolution, related work, citations, results narration, tables, captions, and coherence passes. Do not use for general LaTeX troubleshooting that does not involve the paper's argument or prose.
---

# NLP Paper Writing

Treat the paper as an argument whose claims must be supported by the repository's evidence. Improve the paper without inventing facts, citations, results, or experimental details.

## Establish the task

1. Read `AGENTS.md` and `notes/PAPER_BRIEF.md` when present.
2. Inspect the target section and its immediate neighbors. Do not load the full paper unless the task needs paper-level coherence.
3. Identify whether the user wants discussion, drafting, revision, comment resolution, or a review. If they ask to discuss options first, do not edit yet.
4. Read only the relevant reference below.

When starting a paper and the brief lacks a defensible central claim, do not wait passively for a perfect prompt. Offer or begin a short author interview, unless the user asks to draft immediately.

## Route to focused guidance

- For paper stories, section order, first-time-reader intuition, or a coherence problem that spans paragraphs, read [argument-and-flow.md](references/argument-and-flow.md).
- For paper structure or any section-specific task, read the routing and boundary guidance in [section-guides.md](references/section-guides.md), then load only the relevant module under `references/sections/`.
- For any prose revision or quality pass, read [revision-checklist.md](references/revision-checklist.md).
- Before presenting any draft, revision, or review, always run the mandatory [style-review.md](references/style-review.md) pass. It checks global coherence, internal paragraph structure, sentence-to-sentence flow, and restrained elegance; it is not optional just because the requested edit is local.
- When editing macros or author comments are involved, also read [editing-with-macros.md](references/editing-with-macros.md) and the repository's `EDITING_GUIDE.md` if present.
- When grounding prose in PDFs, notes, prior papers, or citations, read [sources-and-evidence.md](references/sources-and-evidence.md).
- When adding, revising, checking, or formatting citations or related-work claims, read [citations.md](references/citations.md).
- When selecting what belongs in the paper, turning a large technical write-up into a focused draft, choosing headline results, or deciding between main text and appendix, read [editorial-taste.md](references/editorial-taste.md).
- When the user asks to be interviewed, grilled, pressure-tested, or helped to articulate the paper, or when a new paper brief is substantially incomplete, read [author-interview.md](references/author-interview.md).

## Preserve the scientific contract

Before changing prose, state internally:

- the claim being made;
- the evidence that supports it;
- the intended scope and uncertainty;
- the paragraph's role in the section;
- any terminology or approved wording that must remain stable.

Do not strengthen a claim beyond its evidence. Mark missing information with an explicit placeholder or ask the user; never fill gaps with plausible-looking details.

## Work at the requested level

- Choose the revision mode based on the authors' relationship to the draft. For an early draft that the authors have not yet edited themselves, editing macros usually add little value; if the user asks for revisions and their preference is not clear, ask whether they want visible editing macros before editing. If it is clear that the authors have already started working on the draft, or that the prose is approved, use editing macros without asking. Follow `references/editing-with-macros.md` whenever macros are requested or already in use.
- For a new section, agree on its job and claim sequence before polishing sentences when the structure is still uncertain.
- For a local revision, preserve approved prose and change the smallest span that solves the problem.
- For a global pass, first map section and paragraph roles, then revise top-down. Sentence polishing cannot repair a missing argument.
- Treat the style review as a finalization gate: after the requested work is complete, check the whole unit's themes and paragraph roles, trace topic-to-stress links across transitions, and apply elegance only where it improves clarity or emphasis.
- For an important paragraph, consider several distinct versions internally and choose the one that best serves the claim sequence and authorial voice. Do not expose alternatives unless the author asks to compare them.
- Evaluate every subsection from the perspective of a first-time reader. If it mainly previews machinery defined more clearly elsewhere, move or remove it.
- For results and tables, inspect the raw result files and compute or verify values before writing interpretations.
- Before drafting from abundant material, rank it by argumentative value. Do not mirror the source's level of detail or treat every available result as a contribution.

## Deliver a reviewable result

After editing, summarize what changed and flag unresolved scientific or rhetorical decisions. Run an appropriate LaTeX build or structural check when available. Do not silently resolve ambiguity that belongs to the authors.
