# Experimental setup

## Job

Make the reported comparisons interpretable and reproducible. This section owns
the shared experimental substrate; settings unique to one experiment should sit
immediately before that result, and exhaustive records should live in the
appendix.

## Required evidence

Inspect the actual configuration, data documentation, evaluation scripts,
baseline implementations, and compute records. Do not draft setup details from
memory when files are available. Record uncertainty or ask the authors when a
choice cannot be verified.

## Organize by scientific role

- **Research question:** what comparison can the setup answer?
- **Data:** source, split, filtering, sampling, preprocessing, and relevant
  licenses or population limits.
- **Models or systems:** identity, version, scale, access assumptions, and what
  differs across conditions.
- **Training or inference:** budget, optimization, prompts or decoding, seeds,
  stopping rules, and hardware when relevant.
- **Baselines:** what each represents, how it was implemented, and why it is a
  meaningful comparator.
- **Metrics:** definition, direction, aggregation, what question each metric
  answers, and known proxy limitations.
- **Controls:** what is fixed and varied, and which confound each control is
  intended to neutralize.

Do not imply matched tokens, bytes, examples, compute, parameters, context, or
training horizon unless they are actually matched. When a factor cannot be
controlled, state the mismatch and constrain the interpretation.

## Placement

Keep the shared setup compact enough that the reader can retain it. Put local
details beside the corresponding result instead of making readers reconstruct
an experiment from an earlier inventory. Move full hyperparameters, prompts,
annotation instructions, and environment details to a referenced appendix or
artifact without hiding information essential to judging the central claim.

## Source basis

Primarily *Ever Growing Academic Writing*, PDF pages 6 and 23, on serving deep
readers, reproducibility, and separating setup from Method and Results; Vered
Shwartz's *Tips for Writing NLP Papers*, PDF pages 1–5, on abstraction before
technical detail, self-containment, honest comparison, and reporting unfavorable
results; and *Academic Writing for NLP*, PDF pages 25–27, on separating
background, modeling, and result narration.
