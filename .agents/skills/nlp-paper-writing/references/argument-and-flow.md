# Argument, intuition, and flow

Use this guide before drafting a paper story or when local prose edits fail to make a section coherent.

## Build a claim chain

Treat the paper as a sequence of propositions rather than a sequence of topics. Write the argument in atomic form before drafting:

```text
A: accepted premise or established observation
B: additional constraint, cost, or empirical fact
C: conclusion entailed or strongly motivated by A and B
D: the paper's answer to C
E: evidence that establishes D
```

Every major paragraph should add a premise, derive a consequence, define the answer, or provide evidence. If a paragraph only says that a topic exists, it probably does not advance the argument.

Test each inference. When `A + B` does not support `C`, add the missing premise, narrow `C`, or change the order. A citation or transition word cannot supply missing logic.

## Give the reader a concrete scene

Picture the people or systems that experience the problem. For a method paper, useful scenes often include:

- a user encountering the failure or cost;
- a researcher trying to train, evaluate, or extend the system;
- a provider deploying or maintaining it under operational constraints.

Use the scene to choose explanations and examples, not necessarily as a literal narrative in the paper. The prose should make the consequence concrete enough that a reader can reconstruct the scene.

## Separate section roles

Assign each section one sentence-long job. Then ask of every paragraph: does a first-time reader need this information here?

- Background supplies concepts and prior evidence needed before the answer appears.
- Motivation turns the problem into a concrete tension and develops intuition.
- Method introduces the answer and defines its mechanics.
- Experimental Setup states the substrate shared across experiments.
- Results introduce experiment-local setup, report evidence, and interpret it.
- Related Work positions the answer after the reader understands it.

Do not retain a subsection merely because its content is correct. It earns space only if it improves understanding, supports a later inference, or prevents a likely misunderstanding.

## Keep the answer early and stable

The first page should become paper-specific quickly. Avoid an opening paragraph that could introduce any paper about the broad area. Once the paper's answer has been introduced, continue developing that answer and its evidence. Do not return to a long paper-by-paper literature discussion; place any prerequisite prior-work motivation before the answer and reserve detailed positioning for Related Work.

## Draft through alternatives

For consequential paragraphs, privately draft several versions with different organizing choices, such as concrete consequence first, mechanism first, or contrast first. Select one complete paragraph based on logical fit, precision, pacing, and consistency with the author's prose. Avoid stitching together locally polished sentences from incompatible versions.

## End with a consequence

The final paragraph of an introduction or conclusion should do more than state a slogan. Synthesize the established claims and show what the method or finding changes, enables, or makes possible. Generate interest through the consequence itself rather than evaluative adjectives.
