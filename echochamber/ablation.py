#!/usr/bin/env python3
"""
EchoChamber Ablation - evidence-salience studies via repeated debates.

Which piece of evidence actually decides the case? Run the same debate many
times with all evidence (baseline), then again with one document removed at a
time, and measure how the verdict distribution moves. Built for large
campaigns: runs are checkpointed to a manifest and resumable, can execute in
parallel, and every debate is budget-capped with force-verdict on so no run
dies without a ruling.

Two modes:
- Auto (salience evaluator): every evidence file gets a leave-one-out
  scenario, except files the user protects.
- Manual grid: explicit scenarios, each with a name, run count, and the
  set of evidence files to EXCLUDE.

Methodology guards baked in:
- Web search is off (a searching advocate could re-import removed evidence).
- Context strategy is pinned to "full" (auto-RAG retrieves differently per
  condition, confounding the ablation).
- The primary outcome is the judge's signed verdict margin (-100..+100,
  positive favors the prosecution), which has far more statistical power
  per run than win/loss counts.

Usage:
    python -m echochamber.ablation \\
      --case-folder cases/example_case \\
      --topic "Python vs Rust" --position "Rust should be chosen" \\
      --provider gemini --model gemini-2.5-flash \\
      --runs 20 --rounds 2 --parallel 4 --output ./ablation/rust_case
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .core.evidence import EvidenceStore
from .core.runner import DebateSpec, run_debate
from .core.stats import bootstrap_mean_diff, holm_adjust
from .providers import create_provider


@dataclass
class Scenario:
    """One experimental condition: the case minus `exclude`."""
    name: str
    runs: int
    exclude: list = field(default_factory=list)


@dataclass
class AblationConfig:
    """A full ablation campaign."""
    case_folder: str
    topic: str
    position: str
    provider: str = "gemini"
    model: Optional[str] = None
    judge_provider: Optional[str] = None   # defaults to provider
    judge_model: Optional[str] = None      # defaults to model
    rounds: int = 2
    runs: int = 20                          # per condition (auto mode)
    protect: list = field(default_factory=list)   # never ablated (auto mode)
    scenarios: list = field(default_factory=list)  # manual grid; empty = auto
    max_total_tokens_per_debate: Optional[int] = 60_000
    # Campaign-wide hard stop. The scheduler refuses to dispatch new debates
    # once completed runs total this many tokens; with N concurrent debates
    # the worst-case overshoot is N x max_total_tokens_per_debate (in-flight
    # debates still finish under their own per-debate budgets).
    max_campaign_tokens: Optional[int] = None
    max_response_tokens: int = 4096
    output_dir: str = "./ablation"
    parallel: int = 1
    verbose: bool = False

    def fingerprint(self) -> dict:
        """The parts that must not change across a resumed campaign."""
        return {
            "case_folder": self.case_folder,
            "topic": self.topic,
            "position": self.position,
            "provider": self.provider,
            "model": self.model,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "rounds": self.rounds,
        }


def auto_scenarios(
    evidence_names: list, protect: list, runs: int
) -> list[Scenario]:
    """Baseline + one leave-one-out scenario per unprotected evidence file."""
    protected = set(protect)
    scenarios = [Scenario(name="baseline", runs=runs, exclude=[])]
    for name, _section in evidence_names:
        if name in protected:
            continue
        scenarios.append(Scenario(name=f"minus {name}", runs=runs, exclude=[name]))
    return scenarios


def signed_margin(winner: Optional[str], strength: Optional[int]) -> Optional[float]:
    """Verdict as a signed margin: positive favors the prosecution."""
    if strength is None:
        return None
    if winner == "prosecution":
        return float(strength)
    if winner == "defense":
        return -float(strength)
    return 0.0


def _spec_for(config: AblationConfig, scenario: Scenario) -> DebateSpec:
    return DebateSpec(
        topic=config.topic,
        position=config.position,
        prosecution_provider=config.provider,
        prosecution_model=config.model,
        defense_provider=config.provider,
        defense_model=config.model,
        moderator_provider=config.judge_provider or config.provider,
        moderator_model=config.judge_model or config.model,
        max_rounds=config.rounds,
        case_folder=config.case_folder,
        exclude_evidence=list(scenario.exclude),
        # Methodology guards — see module docstring
        enable_search=False,
        context_strategy="full",
        max_total_tokens=config.max_total_tokens_per_debate,
        force_verdict=True,
        max_response_tokens=config.max_response_tokens,
        verbose=config.verbose,
        transcript_dir=str(Path(config.output_dir) / "transcripts"),
    )


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn write from a crashed run; redo that run
    return records


def run_ablation(
    config: AblationConfig,
    provider_factory: Callable = create_provider,
    log: Callable[[str], None] = print,
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Run (or resume) an ablation campaign and return the report dict.

    Every completed debate is appended to <output_dir>/runs.jsonl; rerunning
    with the same output dir skips completed runs, so a crashed or stopped
    campaign continues where it left off.
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "runs.jsonl"
    config_path = out / "config.json"

    # A resumed campaign must be the same experiment
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config.fingerprint():
            raise ValueError(
                f"{config_path} holds a different experiment "
                f"(topic/models/case changed). Use a fresh --output dir."
            )
    else:
        config_path.write_text(json.dumps(config.fingerprint(), indent=2))

    # Resolve evidence and scenarios
    store = EvidenceStore.load(config.case_folder)
    names = store.list_names()
    known = {n for n, _ in names}
    scenarios = config.scenarios or auto_scenarios(names, config.protect, config.runs)
    for scenario in scenarios:
        unknown = [e for e in scenario.exclude if e not in known]
        if unknown:
            raise ValueError(f"Scenario '{scenario.name}' excludes unknown evidence: {unknown}")

    # Reference condition for salience deltas
    reference = next((s for s in scenarios if not s.exclude), scenarios[0])

    records = _load_manifest(manifest_path)
    done = {(r["scenario"], r["run"]) for r in records}
    pending = [
        (scenario, i)
        for scenario in scenarios
        for i in range(scenario.runs)
        if (scenario.name, i) not in done
    ]

    total = sum(s.runs for s in scenarios)
    log(f"Ablation campaign: {len(scenarios)} scenarios x runs = {total} debates "
        f"({len(done)} already complete, {len(pending)} to go, parallel={config.parallel})")

    manifest_lock = threading.Lock()
    progress = {"n": len(done), "tokens": sum(r["tokens"] for r in records)}

    def _campaign_budget_hit() -> bool:
        if not config.max_campaign_tokens:
            return False
        with manifest_lock:
            spent = progress["tokens"]
        if spent >= config.max_campaign_tokens:
            log(f"Campaign token budget reached ({spent:,} >= "
                f"{config.max_campaign_tokens:,}) — no further debates dispatched. "
                f"Raise --max-campaign-tokens and rerun to resume.")
            return True
        return False

    def _run_one(scenario: Scenario, run_idx: int) -> dict:
        spec = _spec_for(config, scenario)
        try:
            outcome = run_debate(
                spec, provider_factory=provider_factory, should_stop=should_stop,
            )
            result = outcome.result
            tin, tout = outcome.usage.total_tokens()
            record = {
                "scenario": scenario.name,
                "run": run_idx,
                "exclude": list(scenario.exclude),
                "winner": result.winner,
                "strength": result.verdict_strength,
                "margin": signed_margin(result.winner, result.verdict_strength),
                "termination": result.termination_reason.value,
                "tokens": tin + tout,
                "cost_usd": round(outcome.usage.total_cost(), 6),
                "transcript": str(outcome.transcript_path or ""),
            }
        except Exception as e:
            record = {
                "scenario": scenario.name,
                "run": run_idx,
                "exclude": list(scenario.exclude),
                "winner": None,
                "strength": None,
                "margin": None,
                "termination": f"error: {str(e)[:200]}",
                "tokens": 0,
                "cost_usd": 0.0,
                "transcript": "",
            }
        with manifest_lock:
            with open(manifest_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            records.append(record)
            progress["n"] += 1
            progress["tokens"] += record["tokens"]
            n = progress["n"]
        log(f"  [{n}/{total}] {scenario.name} run {run_idx}: "
            f"{record['winner'] or 'undecided'}"
            + (f" ({record['margin']:+.0f})" if record["margin"] is not None else "")
            + f" — {record['tokens']:,} tok, ${record['cost_usd']:.4f}")
        if on_progress:
            on_progress(n, total, record)
        return record

    if config.parallel > 1:
        # Dispatch in waves of `parallel` so the campaign budget check runs
        # between waves — bounding overshoot to one wave of per-debate budgets.
        with ThreadPoolExecutor(max_workers=config.parallel) as pool:
            queue = list(pending)
            while queue:
                if (should_stop and should_stop()) or _campaign_budget_hit():
                    break
                wave, queue = queue[:config.parallel], queue[config.parallel:]
                futures = [pool.submit(_run_one, s, i) for s, i in wave]
                for future in as_completed(futures):
                    future.result()
    else:
        for scenario, i in pending:
            if should_stop and should_stop():
                log("Ablation stopped; rerun with the same --output to resume.")
                break
            if _campaign_budget_hit():
                break
            _run_one(scenario, i)

    report = summarize(records, scenarios, reference.name, config)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(format_report(report))
    return report


# Significance level for the salience readings. A document is only labeled
# as moving the verdict when its effect survives both checks below.
ALPHA = 0.05
BOOTSTRAP_SEED = 0
CORRECTION_NOTE = (
    "Readings use a 95% percentile-bootstrap CI on the margin delta "
    "(10,000 resamples, seeded) and Holm-Bonferroni adjusted p-values across "
    "all scenarios in this report; a direction is shown only when the CI "
    "excludes 0 and the adjusted p < 0.05."
)


def summarize(
    records: list, scenarios: list, reference_name: str, config: AblationConfig
) -> dict:
    """Per-scenario stats and salience deltas against the reference.

    Each salience delta gets a bootstrap confidence interval. Because one
    report compares every ablated scenario against the same baseline, the
    p-values are Holm-adjusted across that family before any scenario is
    labeled as helping either side.
    """
    def ok_rows(name: str) -> list:
        return [
            r for r in records
            if r["scenario"] == name and not str(r["termination"]).startswith("error")
        ]

    def margins_for(name: str) -> list:
        return [r["margin"] for r in ok_rows(name) if r["margin"] is not None]

    def stats_for(name: str) -> dict:
        rows = ok_rows(name)
        n = len(rows)
        margins = margins_for(name)
        wins = {"prosecution": 0, "defense": 0, "other": 0}
        for r in rows:
            wins[r["winner"] if r["winner"] in ("prosecution", "defense") else "other"] += 1
        mean = sum(margins) / len(margins) if margins else None
        se = None
        if margins and len(margins) > 1:
            var = sum((m - mean) ** 2 for m in margins) / (len(margins) - 1)
            se = (var / len(margins)) ** 0.5
        return {
            "n": n,
            "prosecution_wins": wins["prosecution"],
            "defense_wins": wins["defense"],
            "draws_or_undecided": wins["other"],
            "prosecution_win_rate": round(wins["prosecution"] / n, 3) if n else None,
            "mean_margin": round(mean, 1) if mean is not None else None,
            "se_margin": round(se, 1) if se is not None else None,
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        }

    per_scenario = {}
    ref = stats_for(reference_name)
    ref_margins = margins_for(reference_name)
    tested = []  # scenario names with a bootstrap p-value, in report order
    for scenario in scenarios:
        s = stats_for(scenario.name)
        if scenario.name != reference_name and s["mean_margin"] is not None and ref["mean_margin"] is not None:
            margins = margins_for(scenario.name)
            s["salience_margin_delta"] = round(
                sum(margins) / len(margins) - sum(ref_margins) / len(ref_margins), 1
            )
            if len(margins) >= 2 and len(ref_margins) >= 2:
                boot = bootstrap_mean_diff(
                    margins, ref_margins,
                    seed=f"{BOOTSTRAP_SEED}:{reference_name}:{scenario.name}",
                )
                s["salience_ci95"] = [round(boot.ci_low, 1), round(boot.ci_high, 1)]
                s["salience_p"] = round(boot.p_value, 4)
                s["_ci_excludes_zero"] = boot.excludes_null()
                tested.append(scenario.name)
            else:
                s["reading"] = "too few runs for an interval"
        if scenario.name != reference_name and s["prosecution_win_rate"] is not None and ref["prosecution_win_rate"] is not None:
            s["salience_win_rate_delta"] = round(
                s["prosecution_win_rate"] - ref["prosecution_win_rate"], 3
            )
        s["exclude"] = list(scenario.exclude)
        per_scenario[scenario.name] = s

    # Holm-Bonferroni across every scenario that got a p-value
    adjusted = holm_adjust([per_scenario[name]["salience_p"] for name in tested])
    for name, p_adj in zip(tested, adjusted):
        s = per_scenario[name]
        s["salience_p_holm"] = round(p_adj, 4)
        significant = s.pop("_ci_excludes_zero") and p_adj < ALPHA
        delta = s["salience_margin_delta"]
        s["reading"] = (
            "no detectable effect" if not significant else
            "removal helps prosecution" if delta > 0 else
            "removal helps defense"
        )

    errors = [r for r in records if str(r["termination"]).startswith("error")]
    return {
        "kind": "echochamber-ablation",
        "case_folder": config.case_folder,
        "topic": config.topic,
        "position": config.position,
        "reference": reference_name,
        "scenarios": per_scenario,
        "multiple_comparisons": {
            "method": "holm-bonferroni",
            "family_size": len(tested),
            "alpha": ALPHA,
            "note": CORRECTION_NOTE,
        },
        "total_debates": len(records),
        "errored_debates": len(errors),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "total_tokens": sum(r["tokens"] for r in records),
    }


def format_report(report: dict) -> str:
    """Markdown salience table."""
    lines = [
        f"# Evidence ablation: {report['topic']}",
        "",
        f"Position: *{report['position']}* — margin > 0 favors the prosecution. "
        f"Salience Δ = scenario margin − `{report['reference']}` margin.",
        "",
        "| Scenario | n | P wins | D wins | mean margin ± SE | salience Δ [95% CI] | p (Holm) | reading |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in report["scenarios"].items():
        margin = (
            f"{s['mean_margin']:+.1f} ± {s['se_margin']:.1f}"
            if s["mean_margin"] is not None and s["se_margin"] is not None
            else (f"{s['mean_margin']:+.1f}" if s["mean_margin"] is not None else "—")
        )
        delta = s.get("salience_margin_delta")
        ci = s.get("salience_ci95")
        delta_str = "—" if delta is None else f"{delta:+.1f}"
        if ci:
            delta_str += f" [{ci[0]:+.1f}, {ci[1]:+.1f}]"
        p_holm = s.get("salience_p_holm")
        lines.append(
            f"| {name} | {s['n']} | {s['prosecution_wins']} | {s['defense_wins']} "
            f"| {margin} | {delta_str} "
            f"| {'—' if p_holm is None else f'{p_holm:.3g}'} "
            f"| {s.get('reading', 'reference')} |"
        )
    mc = report.get("multiple_comparisons")
    lines += [""]
    if mc:
        lines += [
            f"Multiple comparisons: {mc['note']} Family size: {mc['family_size']}.",
            "",
        ]
    lines += [
        f"Total: {report['total_debates']} debates, {report['total_tokens']:,} tokens, "
        f"${report['total_cost_usd']}"
        + (f" · {report['errored_debates']} errored" if report["errored_debates"] else ""),
    ]
    return "\n".join(lines)


def load_scenarios_file(path: str) -> list[Scenario]:
    """Manual grid from JSON: {"scenarios": [{"name", "runs", "exclude"}]}"""
    raw = json.loads(Path(path).read_text())
    return [Scenario(**s) for s in raw["scenarios"]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evidence-salience ablation over repeated debates",
    )
    parser.add_argument("--case-folder", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--runs", type=int, default=20,
                        help="Runs per condition in auto mode (default: 20)")
    parser.add_argument("--protect", nargs="*", default=[],
                        help="Evidence files never ablated in auto mode")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="JSON file with a manual scenario grid")
    parser.add_argument("--max-total-tokens", type=int, default=60_000,
                        help="Budget per debate (default: 60000)")
    parser.add_argument("--max-campaign-tokens", type=int, default=None,
                        help="Hard stop for the whole campaign; scheduling halts "
                             "once completed runs total this many tokens")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output", type=str, default="./ablation/run",
                        help="Output dir: manifest, transcripts, report (resumable)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AblationConfig(
        case_folder=args.case_folder,
        topic=args.topic,
        position=args.position,
        provider=args.provider,
        model=args.model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        rounds=args.rounds,
        runs=args.runs,
        protect=args.protect,
        scenarios=load_scenarios_file(args.scenarios) if args.scenarios else [],
        max_total_tokens_per_debate=args.max_total_tokens,
        max_campaign_tokens=args.max_campaign_tokens,
        output_dir=args.output,
        parallel=args.parallel,
        verbose=args.verbose,
    )
    report = run_ablation(config)
    print()
    print(format_report(report))
    print(f"\nReport written to {Path(args.output) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
