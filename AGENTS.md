# Paper collaboration instructions

## Grounding

- Read `notes/PAPER_BRIEF.md` before drafting or materially restructuring prose.
- Treat `results/raw/` and primary source papers as evidence; never invent citations, numbers, settings, or findings.
- Keep claims calibrated to the evidence and distinguish findings from interpretations.
- Select material by its role in the paper's central argument, not by how much detail happens to exist in the sources. Preserve scientifically material counterevidence even when it complicates the story.
- Before presenting any draft or revision, run the mandatory style review in `.agents/skills/nlp-paper-writing/references/style-review.md` for global coherence, flow, and restrained elegance.

## LaTeX structure

- Keep one numbered section per file under `text/`; appendices live under `text/appendix/`.
- Put reusable tables in `tables/`, figures in `figures/`, raw analysis outputs in `results/raw/`, and parsed papers in `sources/parsed/`.
- The project uses the ACL format pack (version acl-style-files-master) copied at the repository root. Preserve its template and assets unless the venue's official instructions require a deliberate update.
- Build with `latexmk -pdf main.tex` after meaningful LaTeX changes when the toolchain is available.
- Before sharing citation changes, run `python3 .agents/skills/nlp-paper-writing/scripts/check_citations.py .` and inspect BibTeX warnings from the build.

## Collaboration

- Author comments are `\nimrod{...}`, `\nimroda`, `\nimrods`, `\nimrodr` (Nimrod Segev), `\yuval{...}`, `\yuvala`, `\yuvals`, `\yuvalr` (Yuval Reif), `\roy{...}`, `\roya`, `\roys`, `\royr` (Roy Schwartz).
- For author-worked or approved prose, follow `EDITING_GUIDE.md` and express every revision with `\ma`, `\ms`, or minimal-span `\mr` macros. For an early draft the authors have not yet edited themselves, macros are optional; if the user asks for revisions and their preference is unclear, ask whether they want visible macros before editing. If it is clear that the authors have already started working on the draft, use macros without asking.
- If asked to discuss revision options first, do not edit until the authors choose a direction.
- Preserve unresolved comments and existing proposed edits unless the user asks to address or accept them.
