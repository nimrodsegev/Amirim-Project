# Section guidance router

Read this file when outlining a paper or choosing which section-specific
reference to load. For drafting, revision, or review of one section, read only
that section's module and the cross-cutting references required by `SKILL.md`.

## Choose the paper's shape

A common empirical NLP paper uses abstract → introduction → related work or
background → method → experimental setup → results and analysis → limitations
→ conclusion → appendix. This is a menu, not a mandatory template. Analysis,
resource, dataset, position, and theory papers may need different centers of
gravity, and venue requirements take precedence.

Give every retained section a one-sentence job and a short claim sequence. Do
not add a conventional section when its material is clearer elsewhere. In
particular, keep these boundaries visible:

- Motivation makes the problem worth solving; Background supplies prerequisites.
- Method explains the reusable proposal; Experimental Setup records what was
  done in the reported study.
- Results report observations; Analysis tests explanations; Discussion develops
  implications with calibrated speculation.
- Related Work positions the paper; it should not carry concepts required to
  understand the paper.
- Limitations constrain interpretation; the Appendix supports scrutiny and
  reproduction.

## Load the relevant module

- For the title and the stable name of the contribution, read
  [title-and-terminology.md](sections/title-and-terminology.md).
- For the abstract, read [abstract.md](sections/abstract.md).
- For the introduction, read [introduction.md](sections/introduction.md).
- For background, problem setup, or motivation, read
  [background-and-motivation.md](sections/background-and-motivation.md).
- For a method, model, resource, dataset, or formal proposal, read
  [method.md](sections/method.md).
- For data, baselines, metrics, controls, and shared evaluation details, read
  [experimental-setup.md](sections/experimental-setup.md).
- For empirical findings and result narration, read
  [results.md](sections/results.md).
- For ablations, diagnostics, mechanism claims, discussion, or broader
  interpretation, read [analysis-and-discussion.md](sections/analysis-and-discussion.md).
- For literature synthesis and positioning, read
  [related-work.md](sections/related-work.md).
- For plots, tables, diagrams, qualitative examples, or captions, read
  [figures-tables-and-captions.md](sections/figures-tables-and-captions.md).
- For methodological limits, generalization boundaries, societal impact, or
  ethics, read [limitations-and-ethics.md](sections/limitations-and-ethics.md).
- For the conclusion, read [conclusion.md](sections/conclusion.md).
- For supplementary evidence and reproducibility details, read
  [appendix.md](sections/appendix.md).

## Shared source basis

These modules synthesize three archived sources: *Ever Growing Academic
Writing*, Vered Shwartz's *Tips for Writing NLP Papers*, and *Academic Writing
for NLP*. The source notes in each module identify the most relevant PDF pages.
The modules preserve the sources' recurring principles: put memorable claims
early, write for skimming and deeper reading, prefer a journey of understanding
to project chronology, keep old and new work distinguishable, use concrete
examples, and make confidence consistent across the abstract, body, captions,
and limitations.

Where the sources differ, use the rule that best protects scientific clarity
and integrity. In particular, synthesize Related Work rather than listing
citations, give the conclusion a real conceptual job, and treat active venue
instructions as authoritative.
