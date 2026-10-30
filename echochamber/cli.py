#!/usr/bin/env python3
"""
EchoChamber CLI - Multi-LLM Courtroom Debate

Usage:
    python -m echochamber.cli --topic "Topic" --position "Position to argue for"
"""

import argparse
import sys

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads from .env in current directory or parent directories
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

from .core import create_case_folder
from .core.runner import DebateSpec, run_debate
from .core.usage import UsageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description="EchoChamber: Multi-LLM Courtroom Debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic debate using Anthropic for all agents
  python -m echochamber.cli \\
    --topic "Should AI systems be open source?" \\
    --position "AI systems should be open source for safety and transparency"

  # With evidence folder (auto-selects context strategy)
  python -m echochamber.cli \\
    --topic "Contract Dispute" \\
    --position "The defendant breached the contract" \\
    --case-folder ./cases/contract_dispute

  # Force a specific context strategy
  python -m echochamber.cli \\
    --topic "Patent Dispute" \\
    --position "The patent was infringed" \\
    --case-folder ./cases/patent_case \\
    --context-strategy rag

  # Local models via LM Studio
  python -m echochamber.cli \\
    --topic "Tabs vs Spaces" \\
    --position "Tabs are superior to spaces" \\
    --prosecution-provider lmstudio \\
    --defense-provider lmstudio \\
    --moderator-provider lmstudio

  # Create a new case folder
  python -m echochamber.cli --init-case ./cases/my_new_case --topic "My Topic"
        """,
    )

    # Case initialization (alternative mode)
    parser.add_argument(
        "--init-case",
        type=str,
        metavar="PATH",
        help="Create a new case folder with standard structure and exit",
    )

    # Required arguments (unless --init-case)
    parser.add_argument(
        "--topic",
        required=False,  # Made optional for --init-case
        help="The topic to debate",
    )
    parser.add_argument(
        "--position",
        required=False,  # Made optional for --init-case
        help="The position the prosecution will argue for",
    )

    # Evidence and case folder
    parser.add_argument(
        "--case-folder",
        type=str,
        metavar="PATH",
        help="Path to case folder containing evidence (shared/, prosecution/, defense/, moderator/)",
    )

    # Context strategy for large documents
    parser.add_argument(
        "--context-strategy",
        choices=["auto", "full", "summarize", "rag"],
        default="auto",
        help="Strategy for handling large documents: auto (default), full, summarize, or rag",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=100000,
        help="Target max tokens for evidence context (default: 100000)",
    )
    parser.add_argument(
        "--rag-chunks",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query in RAG mode (default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        metavar="PATH",
        help="Directory to cache summaries and embeddings",
    )

    # Web search (enabled by default)
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web search for all agents (search is enabled by default)",
    )
    parser.add_argument(
        "--max-searches-per-turn",
        type=int,
        default=2,
        help="Maximum web searches per party turn (default: 2)",
    )
    parser.add_argument(
        "--moderator-searches-per-turn",
        type=int,
        default=5,
        help="Maximum web searches per moderator turn (default: 5)",
    )

    # Session config
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        metavar="N",
        help="Repeat the debate N times and tally the verdicts (default: 1)",
    )
    parser.add_argument(
        "--allow-concession",
        action="store_true",
        help="Allow parties to concede (default: disabled, only moderator decides)",
    )
    parser.add_argument(
        "--allow-conviction",
        action="store_true",
        help="Allow agents to be convinced by opposing arguments (default: strict adversarial)",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Hard token budget across all agents; the session halts gracefully when crossed (default: unlimited)",
    )
    parser.add_argument(
        "--force-verdict",
        action="store_true",
        help="Cut to the final ruling when remaining budget drops below ~10k tokens, guaranteeing a verdict",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output during debate",
    )

    # Provider configuration (defaults to LM Studio for local development)
    default_provider = "lmstudio"
    provider_choices = ["anthropic", "openai", "together", "lmstudio", "gemini"]

    for role in ("prosecution", "defense", "moderator"):
        parser.add_argument(
            f"--{role}-provider",
            default=default_provider,
            choices=provider_choices,
            help=f"LLM provider for {role} (default: {default_provider})",
        )
        parser.add_argument(
            f"--{role}-model",
            default=None,
            help=f"Model for {role} (defaults to the provider's role default; auto-detected for lmstudio)",
        )
        parser.add_argument(
            f"--{role}-instructions",
            default="",
            metavar="TEXT",
            help=f"Custom instructions appended to the {role} system prompt",
        )

    # Output
    parser.add_argument(
        "--save-transcript",
        type=str,
        help="Path to save transcript JSON",
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        default="./transcripts",
        help="Directory to save markdown transcripts (default: ./transcripts)",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable automatic transcript saving",
    )
    parser.add_argument(
        "--usage-log",
        type=str,
        metavar="PATH",
        help="Append per-call usage events (JSONL, agent-stable event shape)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --init-case mode
    if args.init_case:
        topic = args.topic or "Untitled Case"
        case_path = create_case_folder(args.init_case, topic)
        print(f"Created case folder: {case_path}")
        print("\nFolder structure:")
        print(f"  {case_path}/")
        print("  ├── shared/      # Evidence visible to all parties")
        print("  ├── prosecution/ # Private prosecution evidence")
        print("  ├── defense/     # Private defense evidence")
        print("  ├── moderator/   # Moderator-only guidance")
        print("  └── README.md    # Instructions")
        print("\nSupported files: .txt, .md, .json, .csv, .pdf")
        return

    # Validate required args for debate mode
    if not args.topic or not args.position:
        print("Error: --topic and --position are required for debates")
        print("Use --help for usage information")
        sys.exit(1)

    enable_search = not args.no_search

    spec = DebateSpec(
        topic=args.topic,
        position=args.position,
        prosecution_provider=args.prosecution_provider,
        prosecution_model=args.prosecution_model,
        prosecution_instructions=args.prosecution_instructions,
        defense_provider=args.defense_provider,
        defense_model=args.defense_model,
        defense_instructions=args.defense_instructions,
        moderator_provider=args.moderator_provider,
        moderator_model=args.moderator_model,
        moderator_instructions=args.moderator_instructions,
        max_rounds=args.max_rounds,
        allow_concession=args.allow_concession,
        allow_conviction=args.allow_conviction,
        case_folder=args.case_folder,
        context_strategy=args.context_strategy,
        max_context_tokens=args.max_context_tokens,
        rag_chunks=args.rag_chunks,
        cache_dir=args.cache_dir,
        enable_search=enable_search,
        max_searches_per_turn=args.max_searches_per_turn,
        moderator_searches_per_turn=args.moderator_searches_per_turn,
        max_total_tokens=args.max_total_tokens,
        force_verdict=args.force_verdict,
        verbose=not args.quiet,
        transcript_dir=args.transcript_dir,
        save_transcript_json=args.save_transcript,
        save_transcript_markdown=not args.no_transcript,
        usage_log=args.usage_log,
    )

    print("=" * 60)
    print("ECHOCHAMBER - Multi-LLM Courtroom Debate")
    print("=" * 60)
    print(f"\nTopic: {spec.topic}")
    print(f"Position: {spec.position}")
    print(f"Max rounds: {spec.max_rounds}")
    print(f"Mode: {'Allow conviction' if spec.allow_conviction else 'Strict adversarial'}")
    if enable_search:
        print(f"Web search: Enabled (advocates: {spec.max_searches_per_turn}/turn, moderator: {spec.moderator_searches_per_turn}/turn)")
    else:
        print("Web search: Disabled")
    print()

    iterations = max(1, args.iterations)
    # One shared meter: the token budget and usage log span all iterations
    meter = UsageMeter(hard_limit_tokens=spec.max_total_tokens)
    usage_log = spec.usage_log
    spec.usage_log = None  # written once at the end, not per iteration

    outcomes = []
    try:
        for i in range(1, iterations + 1):
            if iterations > 1:
                print(f"\n{'=' * 60}\nITERATION {i}/{iterations}\n{'=' * 60}")
            outcome = run_debate(spec, meter=meter)
            outcomes.append(outcome)
            if outcome.result.termination_reason.value in ("token_budget", "cancelled"):
                break
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(1)
    except ValueError as e:
        # Provider setup errors (missing API keys etc.)
        print(f"\nError initializing agents: {e}")
        print("\nMake sure you have the required API keys set:")
        print("  - ANTHROPIC_API_KEY for Anthropic")
        print("  - OPENAI_API_KEY for OpenAI")
        print("  - TOGETHER_API_KEY for Together.AI")
        print("  - GEMINI_API_KEY for Google Gemini")
        print("  - LM Studio running on localhost:1234 for local models")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during debate: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    result = outcomes[-1].result

    # Print summary
    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print(f"Rounds completed: {result.rounds_completed}")
    print(f"Termination: {result.termination_reason.value}")
    if len(outcomes) > 1:
        from collections import Counter
        tally = Counter((o.result.winner or "undecided") for o in outcomes)
        print("Verdicts: " + " · ".join(f"{w}: {c}" for w, c in tally.most_common()))
    elif result.winner:
        print(f"Winner: {result.winner.upper()}")
    else:
        print("Winner: UNDECIDED")

    print()
    print(meter.summary())
    if usage_log:
        meter.write_jsonl(usage_log)
        print(f"Usage events appended to: {usage_log}")


if __name__ == "__main__":
    main()
