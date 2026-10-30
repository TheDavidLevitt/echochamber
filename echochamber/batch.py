#!/usr/bin/env python3
"""
EchoChamber Batch Runner - Run multiple debate variations simultaneously.

Runs happen in-process on a thread pool (debate turns are sequential API
calls, so threads waiting on I/O parallelize cleanly across runs).

Usage:
    python -m echochamber.batch --config runs.yaml
    python -m echochamber.batch --topic "Topic" --variations providers
"""

import argparse
import sys
import json
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .core.costs import estimate_run_cost
from .core.runner import (
    DEFAULT_MODELS as DEFAULT_ADVOCATE_MODELS,
    DEFAULT_MODERATOR_MODELS,
    DebateSpec,
    run_debate,
)
from .core.session import TerminationReason


@dataclass
class RunVariation:
    """A single run configuration."""
    name: str
    topic: str
    position: str
    prosecution_provider: str = "lmstudio"
    prosecution_model: Optional[str] = None
    defense_provider: str = "lmstudio"
    defense_model: Optional[str] = None
    moderator_provider: str = "lmstudio"
    moderator_model: Optional[str] = None
    max_rounds: int = 3
    case_folder: Optional[str] = None

    def resolve_models(self):
        """Resolve model names from provider defaults."""
        if not self.prosecution_model:
            self.prosecution_model = DEFAULT_ADVOCATE_MODELS.get(self.prosecution_provider, "unknown")
        if not self.defense_model:
            self.defense_model = DEFAULT_ADVOCATE_MODELS.get(self.defense_provider, "unknown")
        if not self.moderator_model:
            self.moderator_model = DEFAULT_MODERATOR_MODELS.get(self.moderator_provider, "unknown")

    def estimate_cost(self, evidence_tokens: int = 0) -> tuple[float, float, dict]:
        """Estimate the cost of this run."""
        self.resolve_models()
        return estimate_run_cost(
            self.prosecution_model,
            self.defense_model,
            self.moderator_model,
            self.max_rounds,
            evidence_tokens,
        )

    def to_spec(self, output_dir: Path) -> DebateSpec:
        """Convert to a runnable DebateSpec (quiet, transcripts to output_dir)."""
        self.resolve_models()
        return DebateSpec(
            topic=self.topic,
            position=self.position,
            prosecution_provider=self.prosecution_provider,
            prosecution_model=self.prosecution_model,
            defense_provider=self.defense_provider,
            defense_model=self.defense_model,
            moderator_provider=self.moderator_provider,
            moderator_model=self.moderator_model,
            max_rounds=self.max_rounds,
            case_folder=self.case_folder,
            verbose=False,
            transcript_dir=str(output_dir),
        )


def generate_provider_variations(
    topic: str,
    position: str,
    providers: list[str],
    max_rounds: int = 3,
    case_folder: Optional[str] = None,
) -> list[RunVariation]:
    """Generate variations using different providers."""
    variations = []
    for provider in providers:
        variations.append(RunVariation(
            name=f"{provider}-all",
            topic=topic,
            position=position,
            prosecution_provider=provider,
            defense_provider=provider,
            moderator_provider=provider,
            max_rounds=max_rounds,
            case_folder=case_folder,
        ))
    return variations


def generate_moderator_variations(
    topic: str,
    position: str,
    advocate_provider: str,
    moderator_providers: list[str],
    max_rounds: int = 3,
    case_folder: Optional[str] = None,
) -> list[RunVariation]:
    """Generate variations with different moderator providers."""
    variations = []
    for mod_provider in moderator_providers:
        variations.append(RunVariation(
            name=f"{advocate_provider}-{mod_provider}-mod",
            topic=topic,
            position=position,
            prosecution_provider=advocate_provider,
            defense_provider=advocate_provider,
            moderator_provider=mod_provider,
            max_rounds=max_rounds,
            case_folder=case_folder,
        ))
    return variations


def generate_rounds_variations(
    topic: str,
    position: str,
    provider: str,
    rounds_list: list[int],
    case_folder: Optional[str] = None,
) -> list[RunVariation]:
    """Generate variations with different round counts."""
    variations = []
    for rounds in rounds_list:
        variations.append(RunVariation(
            name=f"{provider}-R{rounds}",
            topic=topic,
            position=position,
            prosecution_provider=provider,
            defense_provider=provider,
            moderator_provider=provider,
            max_rounds=rounds,
            case_folder=case_folder,
        ))
    return variations


def load_config(config_path: str) -> list[RunVariation]:
    """Load run variations from a YAML or JSON config file."""
    path = Path(config_path)
    content = path.read_text()

    if path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            print("Error: PyYAML required for YAML configs. Install with: pip install pyyaml")
            sys.exit(1)
    else:
        data = json.loads(content)

    variations = []
    base_config = data.get("base", {})

    for run in data.get("runs", []):
        # Merge base config with run-specific config
        config = {**base_config, **run}
        variations.append(RunVariation(**config))

    return variations


def run_single_variation(variation: RunVariation, output_dir: Path) -> dict:
    """Run a single variation in-process and return the result."""
    start_time = time.time()
    try:
        outcome = run_debate(variation.to_spec(output_dir))
        return {
            "name": variation.name,
            "success": outcome.result.termination_reason != TerminationReason.ERROR,
            "elapsed": time.time() - start_time,
            "winner": outcome.result.winner or "undecided",
            "rounds": outcome.result.rounds_completed,
            "cost_usd": outcome.usage.total_cost(),
            "transcript": str(outcome.transcript_path or ""),
            "error": "",
        }
    except Exception as e:
        return {
            "name": variation.name,
            "success": False,
            "elapsed": time.time() - start_time,
            "winner": "",
            "rounds": 0,
            "cost_usd": 0.0,
            "transcript": "",
            "error": str(e),
        }


def run_batch(
    variations: list[RunVariation],
    output_dir: Path,
    parallel: int = 1,
    cost_threshold: float = 2.0,
    skip_approval: bool = False,
) -> list[dict]:
    """
    Run multiple variations, optionally in parallel.

    Args:
        variations: List of run variations
        output_dir: Directory for transcripts
        parallel: Number of parallel runs (1 = sequential)
        cost_threshold: Prompt for approval if estimated cost exceeds this
        skip_approval: Skip cost approval prompt

    Returns:
        List of result dicts
    """
    # Calculate total estimated cost
    total_min = 0.0
    total_max = 0.0
    print("\n" + "=" * 60)
    print("BATCH RUN COST ESTIMATE")
    print("=" * 60)

    for var in variations:
        min_cost, max_cost, breakdown = var.estimate_cost()
        total_min += min_cost
        total_max += max_cost
        print(f"\n{var.name}:")
        print(f"  ${min_cost:.4f} - ${max_cost:.4f}")
        print(f"  Prosecution: {var.prosecution_provider}/{var.prosecution_model}")
        print(f"  Defense: {var.defense_provider}/{var.defense_model}")
        print(f"  Moderator: {var.moderator_provider}/{var.moderator_model}")

    print("\n" + "-" * 40)
    print(f"TOTAL ESTIMATED: ${total_min:.2f} - ${total_max:.2f}")
    print(f"Number of runs: {len(variations)}")
    print("=" * 60)

    # Check cost threshold
    if not skip_approval and total_max > cost_threshold:
        print(f"\n⚠️  Estimated cost (${total_max:.2f}) exceeds threshold (${cost_threshold:.2f})")
        response = input("Do you want to proceed? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Batch run cancelled.")
            return []

    # Run the variations
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def describe(result: dict) -> str:
        if not result["success"]:
            return f"failed: {result['error'] or 'session error'}"
        return (
            f"winner: {result['winner']}, {result['rounds']} rounds, "
            f"${result['cost_usd']:.4f} actual"
        )

    if parallel > 1:
        print(f"\nRunning {len(variations)} variations ({parallel} in parallel)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_single_variation, var, output_dir): var
                for var in variations
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else "✗"
                print(f"  {status} {result['name']} ({result['elapsed']:.1f}s, {describe(result)})")
    else:
        print(f"\nRunning {len(variations)} variations sequentially...")
        for i, var in enumerate(variations, 1):
            print(f"\n[{i}/{len(variations)}] Running: {var.name}")
            result = run_single_variation(var, output_dir)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"  {status} Completed in {result['elapsed']:.1f}s ({describe(result)})")

    # Summary
    successful = sum(1 for r in results if r["success"])
    total_actual = sum(r["cost_usd"] for r in results)
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE: {successful}/{len(results)} succeeded")
    print(f"Actual cost: ${total_actual:.4f}")
    print("=" * 60)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="EchoChamber Batch Runner - Run multiple debate variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  python -m echochamber.batch --config runs.yaml

  # Generate provider variations
  python -m echochamber.batch \\
    --topic "AI Safety" \\
    --position "AI should be regulated" \\
    --variations providers \\
    --providers anthropic gemini together

  # Generate moderator variations (same advocates, different moderators)
  python -m echochamber.batch \\
    --topic "AI Safety" \\
    --position "AI should be regulated" \\
    --variations moderator \\
    --advocate-provider together \\
    --moderator-providers anthropic gemini together

  # Run with cost threshold
  python -m echochamber.batch --config runs.yaml --cost-threshold 5.0

  # Skip approval prompt
  python -m echochamber.batch --config runs.yaml --yes
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML/JSON config file with run variations",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic for generated variations",
    )
    parser.add_argument(
        "--position",
        type=str,
        help="Prosecution position for generated variations",
    )
    parser.add_argument(
        "--case-folder",
        type=str,
        help="Case folder for all variations",
    )
    parser.add_argument(
        "--variations",
        choices=["providers", "moderator", "rounds"],
        help="Type of variations to generate",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["anthropic", "gemini", "together"],
        help="Providers for 'providers' variation type",
    )
    parser.add_argument(
        "--advocate-provider",
        type=str,
        default="together",
        help="Advocate provider for 'moderator' variation type",
    )
    parser.add_argument(
        "--moderator-providers",
        nargs="+",
        default=["anthropic", "gemini", "together"],
        help="Moderator providers for 'moderator' variation type",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Round counts for 'rounds' variation type",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max rounds for generated variations (default: 3)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel runs (default: 1 = sequential)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./transcripts",
        help="Output directory for transcripts (default: ./transcripts)",
    )
    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=2.0,
        help="Prompt for approval if estimated cost exceeds this (default: $2.00)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip cost approval prompt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    variations = []

    if args.config:
        # Load from config file
        variations = load_config(args.config)
    elif args.variations and args.topic and args.position:
        # Generate variations
        if args.variations == "providers":
            variations = generate_provider_variations(
                args.topic,
                args.position,
                args.providers,
                args.max_rounds,
                args.case_folder,
            )
        elif args.variations == "moderator":
            variations = generate_moderator_variations(
                args.topic,
                args.position,
                args.advocate_provider,
                args.moderator_providers,
                args.max_rounds,
                args.case_folder,
            )
        elif args.variations == "rounds":
            provider = args.providers[0] if args.providers else "lmstudio"
            variations = generate_rounds_variations(
                args.topic,
                args.position,
                provider,
                args.rounds,
                args.case_folder,
            )
    else:
        print("Error: Provide --config or (--topic, --position, --variations)")
        print("Use --help for usage information")
        sys.exit(1)

    if not variations:
        print("No run variations configured.")
        sys.exit(1)

    results = run_batch(
        variations,
        Path(args.output_dir),
        parallel=args.parallel,
        cost_threshold=args.cost_threshold,
        skip_approval=args.yes,
    )

    # Exit with error if any runs failed
    if results and not all(r["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
