# Editing with macros

Use visible editing macros when revising author-worked or approved prose:

- `\ma{...}` for inserted text.
- `\ms{...}` for the exact deleted span.
- `\mr{old}{new}` for the smallest replaced span.
- Do not wrap a sentence or paragraph when only a clause or phrase changed.
- Do not nest editing macros.
- For an early draft that the authors have not yet edited themselves, macros are optional because they usually add little value at that stage. If a revision request does not make the author's preference clear, ask whether to use macros before editing; skip the question when it is clear the authors have already started working on the draft.
- If macros are already in use, preserve that reviewable workflow. An explicit user request about the editing mode takes precedence.
- Address requested author comments in source order. Leave the comment in place and wrap it with `\resolved{...}` when addressed.
- Ignore comments explicitly marked as the author's own TODO, comments in commented-out source, and comments inside already deleted text.
- Do not introduce raw edits outside macros or silently accept earlier proposed edits.
