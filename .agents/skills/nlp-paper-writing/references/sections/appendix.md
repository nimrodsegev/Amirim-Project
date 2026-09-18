# Appendix and supplementary material

## Job

Serve readers who need to reproduce, audit, or deeply understand the work
without interrupting the main argument. The appendix complements the main text;
it does not rescue an underspecified method or hide evidence needed to judge the
central claim.

## What belongs here

- full hyperparameters, prompts, templates, preprocessing, and environment details;
- pseudocode, implementation notes, and formal proofs;
- complete metric definitions, statistical procedures, and annotation protocols;
- additional robustness, ablation, qualitative, and failure-case evidence;
- comprehensive tables or figure grids whose main-text takeaway is already clear;
- dataset documentation, licenses, compute records, and release details.

Supporting results belong when they strengthen scope, mechanism, or
reproducibility. A result that changes the perceived contribution, contradicts a
central claim, or is needed to evaluate the main comparison belongs in the main
text even when space is tight.

## Make it usable

Organize the appendix around reader questions rather than the chronology of the
project. Give each subsection an informative title, define terms that cannot be
recovered locally, and connect every important appendix item from the main text.
State what an extra table or figure establishes instead of presenting an
uninterpreted archive dump.

Verify that main-text pointers, labels, values, and terminology match. Keep raw
outputs in the repository even when the appendix reports only a processed view.

## Source basis

Primarily *Ever Growing Academic Writing*, PDF pages 4–6 and 25–26, on supporting
multiple depths of reading, reproducibility, cutting without losing verification,
and serving the most specialized readers; and Vered Shwartz's *Tips for Writing
NLP Papers*, PDF pages 1–5, on placing technical detail appropriately, keeping
the paper self-contained, selecting a coherent story, and retaining unfavorable
evidence. *Academic Writing for NLP*, PDF pages 25–27, reinforces the separation
of background, method, and results that determines what is supplementary.
