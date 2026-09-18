# Editing with macros

Use visible editing macros for revisions to author-worked or approved prose so that changes remain reviewable in the compiled PDF. They are optional for an early draft that the authors have not yet edited themselves: macros generally add little value before the authors have begun shaping the text. If a user asks for revisions to such a draft and has not made their preference clear, ask whether they want macros before editing. When it is clear that the authors have already started working on the draft, use the macros without asking.

If macros are already in use, preserve that reviewable workflow and do not silently switch to plain edits. An explicit user request about the editing mode takes precedence.

## Macro roles

- `\ma{new text}` inserts new text.
- `\ms{old text}` deletes the exact old span.
- `\mr{old}{new}` replaces the smallest changed span.
- An author comment macro such as `\yuval{...}` remains in place and is wrapped in `\resolved{...}` after it is addressed.

The repository's `EDITING_GUIDE.md` is authoritative if it differs from this summary.

## Minimal-edit rule

Keep shared text outside macros. Prefer:

```tex
The model \mr{is fast}{is competitive}\ma{ on English tasks}.
```

Do not replace a full sentence when only a phrase changed. Do not nest editing macros inside `\ma{}` or the replacement arm of `\mr{}{...}`.

## Resolving comments

1. Search for the requested author's unresolved comment macro.
2. Skip comments explicitly marked as the author's own TODO, commented-out comments, and comments inside text already deleted by a macro.
3. Interpret the comment in the context of the paragraph and paper brief.
4. Make the smallest sufficient macro edit.
5. Wrap the original comment with `\resolved{...}`; do not erase it.
6. Continue in source order unless the user requests a subset.

Do not modify editing-macro definitions or accept/reject existing edits unless the user asks.
