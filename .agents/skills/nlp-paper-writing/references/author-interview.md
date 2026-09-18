# Author interview and pressure test

Use this mode to turn tacit project knowledge into a defensible paper brief. The goal is not to collect every fact; it is to discover the paper's most consequential, supportable argument and the decisions the authors must still make.

## Interaction style

- Ask one high-leverage question at a time, or at most three tightly related questions.
- Start from what the author knows; do not present a long questionnaire.
- After each answer, briefly restate the current understanding, distinguish author statements from your inferences, and ask the next question that most reduces uncertainty.
- Be constructively skeptical. Press on importance, novelty, evidence, alternative explanations, scope, and negative results without becoming adversarial.
- If an answer reveals a contradiction, surface it explicitly instead of quietly harmonizing it.
- Keep unknowns visible. Do not convert guesses into decisions.

Respect the requested interaction: if the user says “just draft,” ask only genuinely blocking questions; if they say “discuss only,” do not edit files.

## Interview arc

Choose the next question adaptively rather than reading this as a fixed script.

### 1. Reader and problem

- Who should care, and what do they currently believe or do?
- What concrete failure, cost, contradiction, or missed capability makes the problem consequential?
- What would remain wrong if this paper did not exist?

### 2. Answer and contribution

- What is the one-sentence answer the paper gives?
- What exists or is known after this work that did not exist or was not known before?
- Which proposed contribution is central, and which are supporting work products?

### 3. Evidence

- What is the strongest result, and what exact claim does it support?
- What is held constant in the decisive comparison?
- Which result is unfavorable, null, or difficult to explain, and how does it limit the claim?
- What evidence is still missing for the desired wording?

### 4. Prior work and novelty

- What are the two or three closest papers, and what precise distinction survives a fair comparison?
- Is the difference a new capability, explanation, setting, scale, or merely an implementation choice?
- What would a skeptical expert say has already been done?

### 5. Editorial center and story

- What should a reader remember a week later?
- Which material is core, supporting, boundary evidence, context, or project process?
- What motivating example can recur through the introduction, method, and analysis?

### 6. Scope and integrity

- Where should the paper deliberately make a weaker claim?
- What population, model family, language, dataset, or condition has not been tested?
- What finding must remain visible even though it complicates the story?

## Checkpoints and stopping condition

After every few answers, provide a compact checkpoint:

- current one-sentence paper;
- proposed editorial center;
- candidate claims and supporting evidence;
- unresolved or contradictory points;
- the next question and why it matters.

Stop the interview when the authors can review a coherent brief containing a consequential problem, central answer, claim-evidence map, closest-work distinction, scope, and an initial story. Then offer to update `notes/PAPER_BRIEF.md` and propose an outline. Do not start polished drafting automatically unless requested.

## Focused variants

- **Introduction interview:** pressure-test reader, tension, answer, evidence preview, and contributions.
- **Results interview:** identify decisive comparisons, controls, anomalies, boundary evidence, and supported interpretations.
- **Revision interview:** ask what the current passage must accomplish, what is wrong with it, and what wording or scientific content is non-negotiable.
- **Related-work interview:** identify axes, closest papers, credit due, and the narrow distinction the paper can defend.
