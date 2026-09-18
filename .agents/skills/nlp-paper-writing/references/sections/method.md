# Method

## Job

Teach the paper's reusable contribution clearly enough that an informed reader
can understand what changes, why it may work, and how to implement or test it.
The Method section explains the proposal in general; Experimental Setup records
the particular choices used in the reported experiments.

## Explain in layers

1. Start with the purpose and intuition before symbols or components.
2. Give a compact map of the method's stages or subcomponents in the order the
   section will explain them.
3. Define inputs, outputs, assumptions, timing, and constraints before use.
4. Specify each conceptual operation with only the formalism needed to remove
   ambiguity.
5. Instantiate the definition on a running example.
6. End each subsection with what the defined component enables or why the next
   component is needed.

A method overview figure is useful when it lets the reader follow one example
through the stages. Use the same labels in the figure, equations, headings, and
prose.

## Keep boundaries clean

- Do not report empirical results in Method.
- Distinguish design choices essential to the contribution from training or
  evaluation settings specific to this study.
- Put exhaustive dimensions, initialization choices, sweep grids, and routine
  engineering details in the appendix unless they determine the claim.
- State important assumptions and failure cases beside the operation they
  constrain; do not postpone all caveats to Limitations.
- Use equations when they expose a relationship more clearly than prose. Define
  every symbol and explain the scientific meaning after the equation.

For datasets, resources, and metrics, replace algorithmic stages with the
corresponding construction pipeline: source selection, inclusion criteria,
annotation or transformation, quality control, output schema, and intended use.
Preserve provenance and licensing facts.

## Final checks

Can a reader reconstruct the method without importing unstated lab knowledge?
Is every component motivated before its details? Could any paragraph be moved
unchanged into Experimental Setup, Results, or Related Work? If so, repair the
section boundary.

## Source basis

Primarily *Ever Growing Academic Writing*, PDF pages 6–8 and 23, on section
previews, examples, method purpose, and the Method/Experimental Setup boundary;
Vered Shwartz's *Tips for Writing NLP Papers*, PDF pages 1–2, on moving from
abstract to concrete and previewing technical subcomponents with a visual; and
*Academic Writing for NLP*, PDF page 26, on keeping modeling separate from
results and using a stable-terminology method figure.
