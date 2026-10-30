# Results

> **Status: not yet run.** This page fixes the experiment *before* any
> numbers exist: the exact commands, the table each one produces, and what
> it will cost. Fixing the method first means the results can't steer it.
> The tables below are placeholders in the exact layout the tools emit.

Two small campaigns, both on Gemini so a single key (or a GCP project with
Vertex AI) covers them:

1. **Evidence ablation** on the bundled example case: does removing any one
   document move the judge's verdict, beyond noise?
2. **Debate eval** with three models (candidate, incumbent, judge): does a
   stronger model win side-swapped debates more often than chance?

## Setup

```bash
uv sync --extra all
cp .env.example .env    # set GEMINI_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT
```

Use a key with a low spend limit. Both commands below also carry hard
token caps.

## 1. Evidence ablation (with confidence intervals)

```bash
uv run python -m echochamber.ablation \
  --case-folder cases/example_case \
  --topic "Python vs Rust for High-Performance Data Processing" \
  --position "Rust should be chosen for the new system" \
  --provider gemini --model gemini-2.5-flash \
  --protect evaluation_criteria.md \
  --runs 20 --rounds 2 --parallel 4 \
  --max-total-tokens 60000 --max-campaign-tokens 4000000 \
  --output ./ablation/example_case
```

The run produces 4 scenarios × 20 runs = 80 debates: the baseline plus one
leave-one-out scenario for each of the three evidence files.
`evaluation_criteria.md` is the judge's rubric, not evidence, so it is
protected from ablation. The campaign checkpoints to
`./ablation/example_case/runs.jsonl`. If it is interrupted, rerun the same
command and it resumes.
The report lands in `report.md` / `report.json` in the same folder.

### Result (placeholder)

| Scenario | n | P wins | D wins | mean margin ± SE | salience Δ [95% CI] | p (Holm) | reading |
|---|---|---|---|---|---|---|---|
| baseline | — | — | — | — | — | — | reference |
| minus case_background.md | — | — | — | — | — | — | — |
| minus performance_benchmarks.md | — | — | — | — | — | — | — |
| minus team_productivity.md | — | — | — | — | — | — | — |

**How to read it.** Margin is the judge's signed verdict strength (−100…+100,
positive favors the prosecution). Salience Δ is a scenario's mean margin minus
the baseline's. The bracket is a 95% percentile-bootstrap interval (10,000
resamples, fixed seed). The p-values are Holm–Bonferroni adjusted across the
three ablated scenarios. A row says "removal helps prosecution/defense" only
when its interval excludes 0 *and* its adjusted p < 0.05. Otherwise it says
"no detectable effect". That means the data can't distinguish the effect from
zero; it does not mean the effect is zero.

**Prediction, written before running:** removing `performance_benchmarks.md`
(the prosecution's private evidence) should push margins toward the defense.
Removing `team_productivity.md` should push them toward the prosecution. The
topic is public, though, and the models know plenty about Rust and Python from
pretraining, so the effects may be small.

## 2. Three-model debate eval

```bash
uv run python -m echochamber.evals \
  --candidate-provider gemini --candidate gemini-2.5-flash \
  --incumbent-provider gemini --incumbent gemini-2.0-flash \
  --judge-provider gemini --judge gemini-2.5-pro \
  --topics 6 --rounds 1 --max-total-tokens 30000 \
  --output eval_report.json
```

The eval runs 6 topics × 2 sides = 12 debates. The judge stays the same
throughout, and the candidate argues each side of every topic once.

### Result (placeholder)

| Outcome | Estimate | 95% CI | Null | p (Holm) | Reading |
|---|---|---|---|---|---|
| Score (wins + ½ draws) | — | — | 0.5 | — | — |
| Mean candidate margin | — | — | 0 | — | — |

Record: —W / —L / —D. (Values come from the `significance` block of
`eval_report.json`; the CLI prints the same numbers.)

The intervals resample whole topics, so each side-swapped pair stays
together. The two p-values are Holm-adjusted as one family. With only 6
topics the intervals will be wide, so expect "no detectable difference"
unless the gap between the models is large.

## Cost

These are estimates from the repo's own pricing table
(`echochamber/data/pricing.json`) and its per-turn token heuristics:

| Campaign | Debates | Est. tokens | Est. cost | Hard cap |
|---|---|---|---|---|
| Ablation (2.5 Flash, 2 rounds) | 80 | ~3.4M | ~$1–3 | 4M tokens for the campaign (+ at most one in-flight wave of 4 × 60k) |
| Eval (2.5 Flash vs 2.0 Flash, 2.5 Pro judge) | 12 | ~0.2M | ~$0.2–0.6 | 30k tokens per debate (≤ 0.36M) |

Gemini 2.5 models also bill "thinking" tokens as output, which the heuristic
doesn't model, so plan for the top of each range. Both tools print actual
metered cost per run and in the final report, so the true spend is recorded
next to the results. Running on Vertex AI bills the GCP project, where
credits apply.
