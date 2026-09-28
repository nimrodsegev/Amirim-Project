#!/usr/bin/env python3
"""Check LaTeX citation keys and BibTeX hygiene in a paper repository."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


CITE_RE = re.compile(
    r"\\(?P<command>cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|citeposs|newcite)"
    r"\s*(?:\[[^\]]*\]\s*){0,2}\{(?P<keys>[^}]*)\}"
)
BIB_RE = re.compile(r"^\s*@(?P<kind>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,", re.MULTILINE)
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{(?P<files>[^}]*)\}")
IGNORED_BIB_KINDS = {"comment", "preamble", "string"}


def latex_without_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def collect_citations(root: Path) -> tuple[dict[str, list[str]], list[str], set[str]]:
    uses: dict[str, list[str]] = defaultdict(list)
    generic: list[str] = []
    bibliography_files: set[str] = set()
    for tex in sorted(root.rglob("*.tex")):
        text = latex_without_comments(tex.read_text(encoding="utf-8", errors="replace"))
        for bibliography_match in BIBLIOGRAPHY_RE.finditer(text):
            bibliography_files.update(
                name.strip() for name in bibliography_match.group("files").split(",") if name.strip()
            )
        for match in CITE_RE.finditer(text):
            command = match.group("command")
            if command == "cite":
                generic.append(str(tex.relative_to(root)))
            for key in match.group("keys").split(","):
                key = key.strip()
                if key and key != "*":
                    uses[key].append(str(tex.relative_to(root)))
    return uses, sorted(set(generic)), bibliography_files


def collect_bibkeys(root: Path) -> tuple[dict[str, list[str]], set[str]]:
    definitions: dict[str, list[str]] = defaultdict(list)
    custom_keys: set[str] = set()
    for bib in sorted(root.rglob("*.bib")):
        text = bib.read_text(encoding="utf-8", errors="replace")
        for match in BIB_RE.finditer(text):
            if match.group("kind").lower() in IGNORED_BIB_KINDS:
                continue
            key = match.group("key")
            definitions[key].append(str(bib.relative_to(root)))
            if bib.name == "custom.bib":
                custom_keys.add(key)
    return definitions, custom_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    citations, generic_files, bibliography_files = collect_citations(root)
    definitions, custom_keys = collect_bibkeys(root)
    missing = sorted(set(citations) - set(definitions))
    duplicates = {key: files for key, files in definitions.items() if len(files) > 1}
    unused_custom = sorted(custom_keys - set(citations))
    missing_bibliography_files = sorted(
        name for name in bibliography_files if not (root / f"{name}.bib").exists()
    )
    missing_bibliography_command = bool(citations) and not bibliography_files

    print(f"Citation keys used: {len(citations)}")
    print(f"BibTeX keys available: {len(definitions)}")
    if missing:
        print("\nUndefined citation keys:")
        for key in missing:
            print(f"- {key}: {', '.join(sorted(set(citations[key])))}")
    if duplicates:
        print("\nDuplicate BibTeX keys:")
        for key, files in sorted(duplicates.items()):
            print(f"- {key}: {', '.join(files)}")
    if generic_files:
        print("\nGeneric \\cite commands (prefer explicit \\citep or \\citet):")
        for path in generic_files:
            print(f"- {path}")
    if missing_bibliography_command:
        print("\nCitations are present but no active \\bibliography{...} command was found.")
    if missing_bibliography_files:
        print("\nBibliography files named in LaTeX but not found:")
        for name in missing_bibliography_files:
            print(f"- {name}.bib")
    if unused_custom:
        print("\nUncited custom.bib entries (review; they may be intentionally staged):")
        for key in unused_custom:
            print(f"- {key}")

    if missing or duplicates or missing_bibliography_command or missing_bibliography_files:
        return 1
    print("\nNo undefined or duplicate citation keys found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
