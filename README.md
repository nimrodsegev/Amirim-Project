# Do Language Models Recognize a Phrase Before Reading It?

Amirim final project (HUJI, CS 68101). Nimrod Segev, advised by Yuval Reif and
Roy Schwartz. Written in ACL format; this repository is synced to Overleaf.

## Build

```bash
latexmk -pdf main.tex
```

`main.tex` is in `[review]` mode, so line numbers and editing macros are visible.
Switch to `final` for a camera-ready build.

## Layout

| Path | Contents |
|---|---|
| `text/` | one file per section; appendices under `text/appendix/` |
| `tables/`, `figures/` | **generated** — see `results/README.md`, do not hand-edit |
| `results/raw/` | analysis outputs from the cluster runs |
| `results/processed/` | `paper_numbers.json`, the single source for every reported value |
| `code/` | the scripts that produced the results and that rebuild the tables and figures |
| `notes/PAPER_BRIEF.md` | the paper's claim–evidence map and open decisions — **read this first** |
| `sources/papers/` | related-work PDFs |
| `custom.bib` | verified references not in the ACL Anthology |
| `anthology-1.bib`, `anthology-2.bib` | ACL Anthology snapshot (2026-09-15) |

## Regenerating tables and figures

```bash
python3 code/make_paper_numbers.py
python3 code/make_tables.py
python3 code/make_figures.py
```

## Checks

```bash
python3 .agents/skills/nlp-paper-writing/scripts/check_citations.py .
```

## State of the draft

First full draft. The argument, section structure and all numbers are in place;
nothing has been author-reviewed yet. Open scientific decisions are listed at the
end of `notes/PAPER_BRIEF.md` — in particular which patchscopes layer budget
should carry the headline comparison, and whether the false-positive adjudication
stays in without human validation.
