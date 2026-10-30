"""Programmatic debate runner — one entry point shared by the CLI and batch.

DebateSpec describes a run; run_debate() wires up evidence, agents, and the
session, then saves transcripts. The CLI is a thin argparse layer over this,
and the batch runner calls it in-process (threads, not subprocesses).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .agent import Agent, Role
from .evidence import EvidenceStore
from .preprocessor import (
    ContextStrategy,
    DocumentPreprocessor,
    PreprocessorConfig,
    auto_select_strategy,
    create_summarizer_from_provider,
)
from .session import CourtSession, SessionConfig, SessionResult
from .transcript import RunConfig
from .usage import UsageMeter
from ..providers import create_provider
from ..roles import get_role_prompt

logger = logging.getLogger(__name__)


# Default models per provider (for advocates: prosecution/defense)
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "together": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "lmstudio": "local-model",  # Will be auto-detected
    "gemini": "gemini-2.5-flash",
}

# Default models for moderator/judge role (can differ from advocates)
DEFAULT_MODERATOR_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "together": "deepseek-ai/DeepSeek-R1",
    "lmstudio": "local-model",
    "gemini": "gemini-2.5-flash",
}


def resolve_model(provider: str, model: Optional[str], role: str = "advocate") -> str:
    """Resolve model name based on provider and role if not explicitly specified."""
    if model:
        return model
    if role == "moderator":
        return DEFAULT_MODERATOR_MODELS.get(provider, "unknown")
    return DEFAULT_MODELS.get(provider, "unknown")


@dataclass
class DebateSpec:
    """Everything needed to run one debate."""
    topic: str
    position: str
    prosecution_provider: str = "lmstudio"
    prosecution_model: Optional[str] = None
    defense_provider: str = "lmstudio"
    defense_model: Optional[str] = None
    moderator_provider: str = "lmstudio"
    moderator_model: Optional[str] = None
    max_rounds: int = 3
    allow_concession: bool = False
    allow_conviction: bool = False
    # Per-role custom instructions, appended to the role's system prompt
    # (combined with any instruction files found in the case folder).
    prosecution_instructions: str = ""
    defense_instructions: str = ""
    moderator_instructions: str = ""
    case_folder: Optional[str] = None
    # Evidence file names to drop from the case (ablation studies)
    exclude_evidence: list = field(default_factory=list)
    context_strategy: str = "auto"  # auto | full | summarize | rag
    max_context_tokens: int = 100_000
    rag_chunks: int = 10
    cache_dir: Optional[str] = None
    enable_search: bool = True
    max_searches_per_turn: int = 2
    moderator_searches_per_turn: int = 5
    max_total_tokens: Optional[int] = None  # hard stop across all agents (None = unlimited)
    # Cut to the final ruling when remaining budget drops below the reserve,
    # so a verdict always lands before the hard budget stops the session.
    force_verdict: bool = False
    # Per-response output cap. Reasoning models (Gemini 2.5, o-series) spend
    # "thinking" tokens from this budget, so 1-2k truncates visible output.
    max_response_tokens: int = 4096
    verbose: bool = True
    transcript_dir: str = "./transcripts"
    save_transcript_json: Optional[str] = None
    save_transcript_markdown: bool = True
    usage_log: Optional[str] = None  # JSONL path for agent-stable usage events


@dataclass
class DebateOutcome:
    """A completed run: session result plus artifacts."""
    result: SessionResult
    usage: UsageMeter
    transcript_path: Optional[Path] = None
    strategy: str = "none"


def _load_evidence(spec: DebateSpec, log: Callable[[str], None]):
    """Load and (if needed) preprocess the case folder.

    Returns:
        (evidence_for_session, raw_evidence, strategy_str)
    """
    raw_evidence = EvidenceStore.load(spec.case_folder, exclude=spec.exclude_evidence)
    if spec.exclude_evidence:
        log(f"\nAblated evidence (excluded): {', '.join(spec.exclude_evidence)}")
    log(f"\n{raw_evidence.summary()}")

    if spec.context_strategy == "auto":
        strategy = auto_select_strategy(
            total_chars=raw_evidence.total_chars(),
            max_context_tokens=spec.max_context_tokens,
            num_documents=len(raw_evidence.shared) + len(raw_evidence.prosecution) +
                          len(raw_evidence.defense) + len(raw_evidence.moderator),
        )
        log(f"\nAuto-selected context strategy: {strategy.value}")
    else:
        strategy = ContextStrategy(spec.context_strategy)
        log(f"\nContext strategy: {strategy.value}")

    if strategy == ContextStrategy.FULL:
        return raw_evidence, raw_evidence, strategy.value

    log(f"Preprocessing evidence ({strategy.value})...")
    preproc_config = PreprocessorConfig(
        strategy=strategy,
        max_tokens=spec.max_context_tokens,
        top_k=spec.rag_chunks,
        cache_dir=Path(spec.cache_dir) if spec.cache_dir else None,
    )

    summarizer = None
    if strategy == ContextStrategy.SUMMARIZE:
        # Use the moderator's provider for summarization
        sum_provider = create_provider(spec.moderator_provider, spec.moderator_model)
        summarizer = create_summarizer_from_provider(sum_provider)

    preprocessor = DocumentPreprocessor(preproc_config, summarizer=summarizer)
    evidence = preprocessor.process_evidence_store(raw_evidence)
    log(evidence.summary())
    return evidence, raw_evidence, strategy.value


def _search_instructions(role: Role, native_tools: bool) -> str:
    """Search guidance appended to the system prompt.

    Providers with native tool support get the self-describing web_search
    tool; the rest are instructed in the [SEARCH:] sentinel protocol.
    """
    if native_tools:
        if role == Role.MODERATOR:
            return """

WEB SEARCH (REQUIRED):
You have a web_search tool. You MUST use it to verify legal or factual claims
and find relevant precedents before making rulings — at least one search per
evaluation. Base your rulings on established principles, not just the
arguments presented."""
        return """

WEB SEARCH:
You have a web_search tool for finding data, studies, or expert opinions that
support your position. Your search budget per turn is limited, so use it
strategically on the most impactful queries."""

    if role == Role.MODERATOR:
        return """

WEB SEARCH (REQUIRED):
You MUST search the web to verify legal claims and find relevant precedents before making rulings.
Include [SEARCH: your query] in your response to perform a search.

IMPORTANT: You are REQUIRED to use at least one search per evaluation to verify the legal basis of arguments.
Example searches:
- [SEARCH: contract law breach of contract deposit refund precedent]
- [SEARCH: implied warranty consumer protection case law]
- [SEARCH: statute of limitations civil claims jurisdiction]

Search results will be provided to you. Base your rulings on established law, not just the arguments presented.
DO NOT skip searching - your rulings must be grounded in verified legal principles."""
    return """

WEB SEARCH:
You can search the web for supporting evidence by including [SEARCH: your query] in your response.
For example: [SEARCH: statistics on remote work productivity 2024]
Search results will be provided to you, and you can reference them in your arguments.
Use searches strategically to find data, studies, or expert opinions that support your position.
Limit searches to the most impactful queries."""


def build_agent(
    name: str,
    role: Role,
    provider: str,
    model: str,
    allow_conviction: bool = False,
    allow_concession: bool = True,
    enable_search: bool = False,
    extra_instructions: str = "",
    meter: Optional[UsageMeter] = None,
    provider_factory: Callable = create_provider,
    max_tokens: int = 4096,
) -> Agent:
    """Create a configured agent for a debate role."""
    llm_provider = provider_factory(provider, model)

    system_prompt = get_role_prompt(
        role,
        allow_conviction=allow_conviction,
        allow_concession=allow_concession,
    )

    if enable_search:
        system_prompt += _search_instructions(role, llm_provider.supports_tools)

    if extra_instructions:
        system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_instructions}"

    return Agent(
        name=name,
        role=role,
        provider=llm_provider,
        system_prompt=system_prompt,
        meter=meter,
        max_tokens=max_tokens,
    )


def run_debate(
    spec: DebateSpec,
    provider_factory: Callable = create_provider,
    meter: Optional[UsageMeter] = None,
    on_turn: Optional[Callable] = None,
    on_status: Optional[Callable] = None,
    on_delta: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DebateOutcome:
    """
    Run one debate end to end: evidence, agents, session, transcripts.

    Args:
        spec: The debate configuration
        provider_factory: create_provider-compatible factory (injectable for tests)
        meter: Optional pre-built UsageMeter (e.g. for live UI polling);
            created from spec.max_total_tokens if omitted
        on_turn: Callback(speaker, role, content) after each recorded turn
        on_status: Callback(stage, agent) as each phase starts
        on_delta: Callback(speaker, role, fragment) streaming turn text live
        should_stop: Polled between phases; True aborts gracefully

    Returns:
        DebateOutcome with the session result, usage meter, and artifacts
    """
    log = print if spec.verbose else (lambda s: None)
    if meter is None:
        meter = UsageMeter(hard_limit_tokens=spec.max_total_tokens)

    # Load evidence if case folder specified
    evidence = None
    raw_evidence = None
    strategy_str = "none"
    if spec.case_folder:
        try:
            evidence, raw_evidence, strategy_str = _load_evidence(spec, log)
        except Exception as e:
            # Deliberately broad: a debate without evidence beats no debate.
            # log() is silent unless verbose, so also warn via logging.
            log(f"\nWarning: Could not load case folder: {e}")
            log("Proceeding without evidence.\n")
            logger.warning(
                "Could not load case folder %s; proceeding without evidence",
                spec.case_folder, exc_info=True,
            )

    # Initialize search tool if enabled
    search_tool = None
    if spec.enable_search:
        from ..tools import WebSearchTool
        search_tool = WebSearchTool(max_results=5)

    # Combine instructions from evidence files (if any) with spec overrides
    spec_instructions = {
        Role.PROSECUTION: spec.prosecution_instructions,
        Role.DEFENSE: spec.defense_instructions,
        Role.MODERATOR: spec.moderator_instructions,
    }
    role_instructions = {}
    for role, custom in spec_instructions.items():
        from_evidence = raw_evidence.get_instructions_for_role(role) if raw_evidence else ""
        role_instructions[role] = "\n\n".join(
            part.strip() for part in (from_evidence, custom) if part and part.strip()
        )

    # Resolve model names (auto-detect for lmstudio, role-specific defaults)
    prosecution_model = resolve_model(spec.prosecution_provider, spec.prosecution_model, "advocate")
    defense_model = resolve_model(spec.defense_provider, spec.defense_model, "advocate")
    moderator_model = resolve_model(spec.moderator_provider, spec.moderator_model, "moderator")

    log("Initializing agents...")
    prosecution = build_agent(
        name="Prosecution",
        role=Role.PROSECUTION,
        provider=spec.prosecution_provider,
        model=prosecution_model,
        allow_conviction=spec.allow_conviction,
        allow_concession=spec.allow_concession,
        enable_search=spec.enable_search,
        extra_instructions=role_instructions[Role.PROSECUTION],
        meter=meter,
        provider_factory=provider_factory,
        max_tokens=spec.max_response_tokens,
    )
    log(f"  ✓ Prosecution: {prosecution}")

    defense = build_agent(
        name="Defense",
        role=Role.DEFENSE,
        provider=spec.defense_provider,
        model=defense_model,
        allow_conviction=spec.allow_conviction,
        allow_concession=spec.allow_concession,
        enable_search=spec.enable_search,
        extra_instructions=role_instructions[Role.DEFENSE],
        meter=meter,
        provider_factory=provider_factory,
        max_tokens=spec.max_response_tokens,
    )
    log(f"  ✓ Defense: {defense}")

    moderator_can_search = spec.enable_search and spec.moderator_searches_per_turn > 0
    moderator = build_agent(
        name="Moderator",
        role=Role.MODERATOR,
        provider=spec.moderator_provider,
        model=moderator_model,
        allow_conviction=False,  # Moderator doesn't need conviction mode
        allow_concession=False,
        enable_search=moderator_can_search,
        extra_instructions=role_instructions[Role.MODERATOR],
        meter=meter,
        provider_factory=provider_factory,
        max_tokens=spec.max_response_tokens,
    )
    log(f"  ✓ Moderator: {moderator}")

    config = SessionConfig(
        max_rounds=spec.max_rounds,
        allow_concession=spec.allow_concession,
        allow_conviction=spec.allow_conviction,
        force_verdict=spec.force_verdict,
        verbose=spec.verbose,
    )

    run_config = RunConfig(
        topic=spec.topic,
        position=spec.position,
        prosecution_provider=spec.prosecution_provider,
        prosecution_model=getattr(prosecution.provider, "model", prosecution_model),
        defense_provider=spec.defense_provider,
        defense_model=getattr(defense.provider, "model", defense_model),
        moderator_provider=spec.moderator_provider,
        moderator_model=getattr(moderator.provider, "model", moderator_model),
        max_rounds=spec.max_rounds,
        allow_concession=spec.allow_concession,
        allow_conviction=spec.allow_conviction,
        context_strategy=strategy_str,
        enable_search=spec.enable_search,
        case_folder=spec.case_folder or "",
    )

    session = CourtSession(
        prosecution=prosecution,
        defense=defense,
        moderator=moderator,
        config=config,
        evidence=evidence,
        on_turn=on_turn,
        on_status=on_status,
        on_delta=on_delta,
        should_stop=should_stop,
        search_tool=search_tool,
        max_searches_per_turn=spec.max_searches_per_turn if spec.enable_search else 0,
        max_moderator_searches=spec.moderator_searches_per_turn if spec.enable_search else 0,
    )

    result = session.run(topic=spec.topic, prosecution_position=spec.position)
    result.transcript.run_config = run_config

    # Save artifacts
    transcript_path = None
    if spec.save_transcript_markdown:
        transcript_path = result.transcript.save_markdown(spec.transcript_dir)
        log(f"\nTranscript saved to: {transcript_path}")
    if spec.save_transcript_json:
        json_path = Path(spec.save_transcript_json)
        result.transcript.save(json_path)
        log(f"JSON transcript saved to: {json_path}")
    if spec.usage_log:
        meter.write_jsonl(spec.usage_log)
        log(f"Usage events appended to: {spec.usage_log}")

    return DebateOutcome(
        result=result,
        usage=meter,
        transcript_path=transcript_path,
        strategy=strategy_str,
    )
