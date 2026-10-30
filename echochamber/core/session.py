"""Court session orchestrator."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Union

from .agent import Agent, Role
from .transcript import Transcript
from .evidence import EvidenceStore
from .preprocessor import ProcessedEvidenceStore, ContextStrategy
from .turns import run_agent_turn
from .usage import TokenBudgetExceeded
from ..providers import Message, ToolDef

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """Why the session ended."""
    MAX_ROUNDS = "max_rounds"
    CONCESSION = "concession"
    MODERATOR_DECISION = "moderator_decision"
    TOKEN_BUDGET = "token_budget"
    FORCED_VERDICT = "forced_verdict"  # debate cut short to reserve budget for the ruling
    CANCELLED = "cancelled"
    ERROR = "error"


# Tokens reserved for the final ruling when force_verdict is on. Observed
# ruling calls (history + structured verdict, thinking models included) run
# 5-8k tokens on Gemini 2.5 Flash / Claude Haiku; 10k gives headroom.
FORCE_VERDICT_RESERVE_TOKENS = 10_000

# Extra instruction for a ruling after a forced cut-off.
FORCED_VERDICT_NOTE = (
    "The debate was cut short to preserve token budget for judgment. If the "
    "final turn introduced new arguments the opposing side never had a chance "
    "to rebut, disregard them; judge only on the fully-argued record."
)


class SessionCancelled(RuntimeError):
    """Raised internally when should_stop() requests an abort."""


@dataclass
class SessionConfig:
    """Configuration for a court session."""
    max_rounds: int = 5
    allow_concession: bool = True
    allow_conviction: bool = False  # If True, agents can be convinced by opposing arguments
    require_moderator_approval: bool = True
    # Stop debating (and rule immediately) when the remaining token budget
    # drops below FORCE_VERDICT_RESERVE_TOKENS — guarantees a verdict lands
    # before the hard budget kills the session.
    force_verdict: bool = False
    verbose: bool = True


@dataclass
class SessionResult:
    """Result of a completed court session."""
    transcript: Transcript
    termination_reason: TerminationReason
    winner: Optional[str] = None
    rounds_completed: int = 0
    # Judge-reported margin of victory (0-100) from the structured verdict;
    # None on the text protocol. Gives ablation/eval studies a continuous
    # outcome with far more statistical power than win/loss alone.
    verdict_strength: Optional[int] = None


# Structured decision tools for the moderator. Providers without tool
# support fall back to the CONTINUE:/WINNER:/FINAL VERDICT: text protocol.
EVALUATION_TOOL = ToolDef(
    name="submit_evaluation",
    description="Submit your evaluation of the round that just concluded.",
    parameters={
        "type": "object",
        "properties": {
            "continue_debate": {
                "type": "boolean",
                "description": (
                    "False only if one side has clearly won, both sides are "
                    "repeating themselves, or the debate has reached a natural "
                    "conclusion."
                ),
            },
            "winner": {
                "type": "string",
                "enum": ["prosecution", "defense", "none"],
                "description": "Side with the stronger case so far; none if too close to call.",
            },
            "reasoning": {
                "type": "string",
                "description": "Your evaluation of the arguments presented this round.",
            },
        },
        "required": ["continue_debate", "winner", "reasoning"],
    },
)

VERDICT_TOOL = ToolDef(
    name="submit_verdict",
    description="Submit your final ruling for the debate.",
    parameters={
        "type": "object",
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["prosecution", "defense", "draw"],
                "description": "The side that made the stronger case, or draw.",
            },
            "strength": {
                "type": "integer",
                "description": (
                    "Margin of victory, 0-100: 0 = coin flip / draw, "
                    "100 = complete rout. Judge the gap between the cases, "
                    "not your confidence in the topic itself."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Your complete final ruling and summary of the proceedings.",
            },
        },
        "required": ["winner", "strength", "reasoning"],
    },
)


class CourtSession:
    """
    Orchestrates a courtroom debate between LLM agents.

    The session alternates between prosecution and defense,
    with the moderator evaluating after each round.
    """

    def __init__(
        self,
        prosecution: Agent,
        defense: Agent,
        moderator: Agent,
        config: Optional[SessionConfig] = None,
        evidence: Optional[Union[EvidenceStore, ProcessedEvidenceStore]] = None,
        on_turn: Optional[Callable[[str, str, str], None]] = None,
        on_status: Optional[Callable[[str, "Agent"], None]] = None,
        on_delta: Optional[Callable[[str, str, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        search_tool: Optional[object] = None,
        max_searches_per_turn: int = 0,
        max_moderator_searches: int = 0,
    ):
        """
        Initialize a court session.

        Args:
            prosecution: Agent arguing for the proposition
            defense: Agent arguing against
            moderator: Agent controlling flow and deciding outcome
            config: Session configuration
            evidence: Optional evidence store (raw or processed) with case documents
            on_turn: Callback(speaker, role, content) after each turn
            on_status: Callback(stage, agent) fired as each phase starts —
                stages: "opening", "round N: prosecution/defense",
                "round N: evaluation", "final ruling"
            on_delta: Callback(speaker, role, fragment) streaming text as it
                generates (text-only turns; tool-using turns arrive whole)
            should_stop: Polled before each phase; returning True aborts the
                session gracefully (termination reason "cancelled")
            search_tool: Optional WebSearchTool available to agents during turns
            max_searches_per_turn: Search budget per advocate turn (0 = disabled)
            max_moderator_searches: Search budget for the moderator (0 = disabled)
        """
        if prosecution.role != Role.PROSECUTION:
            raise ValueError("Prosecution agent must have PROSECUTION role")
        if defense.role != Role.DEFENSE:
            raise ValueError("Defense agent must have DEFENSE role")
        if moderator.role != Role.MODERATOR:
            raise ValueError("Moderator agent must have MODERATOR role")

        self.prosecution = prosecution
        self.defense = defense
        self.moderator = moderator
        self.config = config or SessionConfig()
        self.evidence = evidence
        self.on_turn = on_turn
        self.on_status = on_status
        self.on_delta = on_delta
        self.should_stop = should_stop
        self.search_tool = search_tool
        self.max_searches_per_turn = max_searches_per_turn
        self.max_moderator_searches = max_moderator_searches

        # Track if using processed evidence for RAG queries
        self._is_processed = isinstance(evidence, ProcessedEvidenceStore)
        self._use_rag = (
            self._is_processed and
            evidence.config.strategy == ContextStrategy.RAG
        )

        self.transcript: Optional[Transcript] = None
        self._running = False

    def run(self, topic: str, prosecution_position: str) -> SessionResult:
        """
        Run the full court session.

        Args:
            topic: The subject of debate
            prosecution_position: The position the prosecution will argue for

        Returns:
            SessionResult with transcript and outcome
        """
        self._running = True
        self.transcript = Transcript(case_topic=topic)

        # Opening context for all parties
        case_context = f"""
CASE TOPIC: {topic}

PROSECUTION POSITION: {prosecution_position}

DEFENSE POSITION: Argue against the prosecution's position.

The debate will proceed in rounds. Each round:
1. Prosecution presents arguments
2. Defense responds
3. Moderator evaluates

The debate ends when:
- Maximum rounds ({self.config.max_rounds}) reached
- A party concedes
- The moderator decides further debate is unnecessary
"""

        # Add shared evidence context if available
        shared_evidence = ""
        if self.evidence:
            shared_evidence = self.evidence.get_shared_context()
            if shared_evidence:
                case_context += f"\n{shared_evidence}"

        self._log(f"Starting court session on: {topic}")
        self._log(f"Prosecution argues: {prosecution_position}")
        if self.evidence:
            self._log(self.evidence.summary())
        self._log("-" * 50)

        round_num = 0
        termination_reason = TerminationReason.MAX_ROUNDS
        winner = None

        try:
            # Moderator opens the session
            self._phase("opening", self.moderator)
            opening = self._get_moderator_opening(case_context)
            self._record_turn(self.moderator.name, "moderator", opening)

            for round_num in range(1, self.config.max_rounds + 1):
                self.transcript.total_rounds = round_num
                self._log(f"\n=== ROUND {round_num} ===\n")

                if self._should_force_verdict():
                    termination_reason = TerminationReason.FORCED_VERDICT
                    break

                # Prosecution argues
                self._phase(f"round {round_num}: prosecution", self.prosecution)
                prosecution_response = self._get_advocate_argument(
                    self.prosecution, case_context, round_num
                )
                self._record_turn(
                    self.prosecution.name, "prosecution", prosecution_response, round_num
                )

                # Check for concession
                if self.config.allow_concession and self._check_concession(prosecution_response):
                    termination_reason = TerminationReason.CONCESSION
                    winner = "defense"
                    break

                if self._should_force_verdict():
                    termination_reason = TerminationReason.FORCED_VERDICT
                    break

                # Defense responds
                self._phase(f"round {round_num}: defense", self.defense)
                defense_response = self._get_advocate_argument(
                    self.defense, case_context, round_num
                )
                self._record_turn(
                    self.defense.name, "defense", defense_response, round_num
                )

                # Check for concession
                if self.config.allow_concession and self._check_concession(defense_response):
                    termination_reason = TerminationReason.CONCESSION
                    winner = "prosecution"
                    break

                if self._should_force_verdict():
                    termination_reason = TerminationReason.FORCED_VERDICT
                    break

                # Moderator evaluates
                if self.config.require_moderator_approval:
                    self._phase(f"round {round_num}: evaluation", self.moderator)
                    should_continue, mod_winner, reasoning = self._get_moderator_evaluation(
                        case_context, round_num
                    )
                    self._record_turn(
                        self.moderator.name, "moderator", reasoning, round_num
                    )

                    if not should_continue:
                        termination_reason = TerminationReason.MODERATOR_DECISION
                        winner = mod_winner
                        break

        except TokenBudgetExceeded as e:
            self._log(f"Token budget reached: {e}")
            termination_reason = TerminationReason.TOKEN_BUDGET
        except SessionCancelled:
            self._log("Session cancelled by user.")
            termination_reason = TerminationReason.CANCELLED
        except Exception as e:
            # Top-level session guard: any failure ends the debate with an
            # ERROR termination and a transcript, instead of losing the run.
            self._log(f"Error during session: {e}")
            logger.warning("Session ended by error", exc_info=True)
            termination_reason = TerminationReason.ERROR
            self.transcript.add_entry(
                speaker="SYSTEM",
                role="system",
                content=f"Session terminated due to error: {e}",
            )

        # Get final ruling (may update winner if not already set). Skipped for
        # budget/cancel stops — the whole point is to stop spending tokens.
        verdict_strength = None
        if termination_reason in (TerminationReason.TOKEN_BUDGET, TerminationReason.CANCELLED):
            final_ruling = (
                f"Session halted ({termination_reason.value}) before a final ruling could be made."
            )
            final_winner = winner
        else:
            self._phase("final ruling", self.moderator)
            try:
                final_ruling, final_winner, verdict_strength = self._get_final_ruling(
                    case_context, termination_reason, winner
                )
            except TokenBudgetExceeded as e:
                self._log(f"Token budget reached during final ruling: {e}")
                termination_reason = TerminationReason.TOKEN_BUDGET
                final_ruling = "Session halted (token_budget) during the final ruling."
                final_winner = winner
        self._record_turn(self.moderator.name, "moderator", final_ruling)

        # Use updated winner from final ruling if available
        if final_winner and not winner:
            winner = final_winner

        # Finalize transcript
        outcome_str = f"{termination_reason.value}"
        if winner:
            outcome_str += f" - {winner} wins"
        self.transcript.finalize(outcome=outcome_str, winner=winner)

        self._running = False

        return SessionResult(
            transcript=self.transcript,
            termination_reason=termination_reason,
            winner=winner,
            rounds_completed=round_num,
            verdict_strength=verdict_strength,
        )

    def _get_evidence_context_for_role(
        self,
        role: Role,
        base_context: str,
        query: Optional[str] = None,
    ) -> str:
        """
        Get case context with role-specific evidence appended.

        For RAG mode, uses query to retrieve relevant chunks.
        """
        if not self.evidence:
            return base_context

        if self._is_processed:
            # ProcessedEvidenceStore
            role_str = role.value if hasattr(role, 'value') else str(role)
            if self._use_rag and query:
                role_evidence = self.evidence.get_context_for_role(role_str, query=query)
            else:
                role_evidence = self.evidence.get_context_for_role(role_str)
        else:
            # Raw EvidenceStore
            role_evidence = self.evidence.get_context_for_role(role)

        if role_evidence:
            return f"{base_context}\n{role_evidence}"
        return base_context

    def _delta_for(self, agent: Agent) -> Optional[Callable[[str], None]]:
        """Bind the on_delta callback to a specific speaker."""
        if not self.on_delta:
            return None
        return lambda fragment: self.on_delta(agent.name, agent.role.value, fragment)

    def _get_moderator_opening(self, case_context: str) -> str:
        """Get the moderator's opening statement."""
        context = self._get_evidence_context_for_role(Role.MODERATOR, case_context)
        messages = [
            Message(
                role="user",
                content=f"{context}\n\nPlease open this court session with a brief statement about the case and rules.",
            )
        ]
        return self.moderator.respond_full(
            messages, on_delta=self._delta_for(self.moderator)
        ).content

    def _get_advocate_argument(self, agent: Agent, case_context: str, round_num: int) -> str:
        """Get an advocate's argument for this round, searches included."""
        history = self.transcript.get_conversation_history()

        # For RAG, use recent context as query
        rag_query = None
        if self._use_rag and history:
            # Use last few exchanges as context for retrieval
            recent = [m.content for m in history[-4:]]
            rag_query = " ".join(recent)[:1000]

        context = self._get_evidence_context_for_role(agent.role, case_context, query=rag_query)

        concession_note = ""
        if self.config.allow_concession:
            concession_note = '\nIf you believe your position is untenable, you may concede by explicitly stating "I CONCEDE" in your response.'

        if agent.role == Role.PROSECUTION:
            turn_note = (
                "This is your opening argument. Establish your main points."
                if round_num == 1
                else "Build on your previous arguments and respond to the defense's points."
            )
            action = "Present your arguments for your position."
        else:
            turn_note = (
                "This is your opening response. Address the prosecution's main points and establish your defense."
                if round_num == 1
                else "Continue your defense and address new points raised by prosecution."
            )
            action = "Present your counter-arguments."

        instruction = f"""
{context}

This is round {round_num}. {action}
{turn_note}
{concession_note}
"""
        messages = history + [Message(role="user", content=instruction)]
        return run_agent_turn(
            agent,
            messages,
            search_tool=self.search_tool,
            max_searches=self.max_searches_per_turn,
            log=self._log,
            on_delta=self._delta_for(agent),
        )

    def _get_moderator_evaluation(
        self, case_context: str, round_num: int
    ) -> tuple[bool, Optional[str], str]:
        """
        Get moderator's evaluation of the round.

        Returns:
            (should_continue, winner_if_decided, reasoning)
        """
        context = self._get_evidence_context_for_role(Role.MODERATOR, case_context)
        history = self.transcript.get_conversation_history()

        if self.moderator.provider.supports_tools:
            result = self._structured_evaluation(context, history, round_num)
            if result is not None:
                return result

        # Text protocol fallback
        instruction = f"""
{context}

Round {round_num} has concluded. Evaluate the arguments presented.

You must respond in this exact format:
CONTINUE: [YES/NO]
WINNER: [PROSECUTION/DEFENSE/NONE]
REASONING: [Your evaluation of the arguments]

Set CONTINUE to NO only if:
- One side has clearly won and further debate would be pointless
- Both sides are repeating arguments without progress
- The debate has reached a natural conclusion

Set WINNER to the side that has made the stronger case so far, or NONE if it's too close to call.
"""
        messages = history + [Message(role="user", content=instruction)]
        response = self.moderator.respond(messages)

        # Parse the response
        should_continue = True
        winner = None

        if "CONTINUE: NO" in response.upper() or "CONTINUE:NO" in response.upper():
            should_continue = False

        if "WINNER: PROSECUTION" in response.upper() or "WINNER:PROSECUTION" in response.upper():
            winner = "prosecution"
        elif "WINNER: DEFENSE" in response.upper() or "WINNER:DEFENSE" in response.upper():
            winner = "defense"

        return should_continue, winner, response

    def _structured_evaluation(
        self, context: str, history: list[Message], round_num: int
    ) -> Optional[tuple[bool, Optional[str], str]]:
        """Round evaluation via the submit_evaluation tool. None on failure."""
        instruction = f"""
{context}

Round {round_num} has concluded. Evaluate the arguments presented, then
submit your evaluation with the submit_evaluation tool.

End the debate (continue_debate = false) only if:
- One side has clearly won and further debate would be pointless
- Both sides are repeating arguments without progress
- The debate has reached a natural conclusion
"""
        messages = history + [Message(role="user", content=instruction)]
        try:
            response = self.moderator.respond_full(
                messages, tools=[EVALUATION_TOOL], tool_choice="submit_evaluation"
            )
        except TokenBudgetExceeded:
            raise  # a budget stop must not trigger the (token-spending) fallback
        except Exception as e:
            # Deliberately broad: any failure of the tool path degrades to the
            # text protocol (see turns.run_agent_turn).
            self._log(f"  [structured evaluation failed ({e}); using text protocol]")
            logger.info("Structured evaluation failed; using text protocol", exc_info=True)
            return None

        call = next(
            (tc for tc in response.tool_calls if tc.name == "submit_evaluation"), None
        )
        if not call:
            return None

        winner = call.arguments.get("winner")
        if winner not in ("prosecution", "defense"):
            winner = None
        reasoning = str(call.arguments.get("reasoning", "")) or response.content
        return bool(call.arguments.get("continue_debate", True)), winner, reasoning

    def _get_final_ruling(
        self,
        case_context: str,
        termination_reason: TerminationReason,
        winner: Optional[str],
    ) -> tuple[str, Optional[str], Optional[int]]:
        """
        Get the moderator's final ruling.

        If search is enabled, allows the moderator to perform searches before ruling.

        Returns:
            (ruling_text, winner, strength) — winner may be updated from the
            ruling; strength is the judge's 0-100 margin (None on text protocol)
        """
        context = self._get_evidence_context_for_role(Role.MODERATOR, case_context)
        history = self.transcript.get_conversation_history()

        if self.moderator.provider.supports_tools:
            structured = self._structured_ruling(
                context, history, termination_reason, winner
            )
            if structured is not None:
                return structured

        # Text protocol fallback
        if winner:
            winner_instruction = f"Winner: {winner}"
        else:
            winner_instruction = """No winner has been determined yet. You must now decide.

In your ruling, you MUST include one of these exact phrases:
- "FINAL VERDICT: PROSECUTION WINS" if prosecution made the stronger case
- "FINAL VERDICT: DEFENSE WINS" if defense made the stronger case
- "FINAL VERDICT: DRAW" if neither side clearly prevailed"""

        # Add search reminder if search is enabled
        search_reminder = ""
        if self.search_tool and self.max_moderator_searches > 0:
            search_reminder = """

IMPORTANT: Before making your final ruling, you SHOULD use web search to verify legal claims.
Include [SEARCH: your query] to search for relevant laws, precedents, or legal analysis.
Your ruling should be grounded in verified legal principles."""

        forced_note = (
            f"\n{FORCED_VERDICT_NOTE}\n"
            if termination_reason == TerminationReason.FORCED_VERDICT
            else ""
        )
        instruction = f"""
{context}

The debate has concluded.
Termination reason: {termination_reason.value}
{winner_instruction}
{forced_note}{search_reminder}
Please provide your final ruling and summary of the proceedings.
"""
        messages = history + [Message(role="user", content=instruction)]
        ruling = self.moderator.respond(messages)

        # Process any search requests in the ruling
        if self.search_tool and self.max_moderator_searches > 0:
            ruling = self._process_ruling_searches(ruling, messages)

        return ruling, self._extract_winner_from_ruling(ruling, winner), None

    def _structured_ruling(
        self,
        context: str,
        history: list[Message],
        termination_reason: TerminationReason,
        winner: Optional[str],
    ) -> Optional[tuple[str, Optional[str], Optional[int]]]:
        """Final ruling via native tools as (ruling, winner, strength). None on failure.

        The moderator may search the web (within budget) and must close with
        the submit_verdict tool.
        """
        from ..tools.search import WEB_SEARCH_TOOL

        can_search = self.search_tool is not None and self.max_moderator_searches > 0
        winner_note = (
            f"A winner has already been determined: {winner}. Your ruling should explain the outcome."
            if winner
            else "No winner has been determined yet — your verdict decides."
        )
        search_note = (
            " Verify key legal or factual claims with the web_search tool before ruling."
            if can_search
            else ""
        )
        forced_note = (
            f" {FORCED_VERDICT_NOTE}"
            if termination_reason == TerminationReason.FORCED_VERDICT
            else ""
        )
        instruction = f"""
{context}

The debate has concluded (reason: {termination_reason.value}). {winner_note}{forced_note}{search_note}
When ready, submit your complete final ruling with the submit_verdict tool.
"""
        convo = history + [Message(role="user", content=instruction)]
        searches_used = 0

        try:
            for _ in range(self.max_moderator_searches + 2):
                if can_search and searches_used < self.max_moderator_searches:
                    tools, tool_choice = [WEB_SEARCH_TOOL, VERDICT_TOOL], None
                else:
                    tools, tool_choice = [VERDICT_TOOL], "submit_verdict"
                response = self.moderator.respond_full(
                    convo, tools=tools, tool_choice=tool_choice
                )

                verdict_call = next(
                    (tc for tc in response.tool_calls if tc.name == "submit_verdict"),
                    None,
                )
                if verdict_call:
                    v = verdict_call.arguments.get("winner")
                    final_winner = winner or (
                        v if v in ("prosecution", "defense", "draw") else None
                    )
                    ruling = str(verdict_call.arguments.get("reasoning", "")) or response.content
                    try:
                        strength = int(verdict_call.arguments.get("strength"))
                        strength = max(0, min(100, strength))
                    except (TypeError, ValueError):
                        strength = None
                    return ruling, final_winner, strength

                if not response.tool_calls:
                    # Plain text ruling despite the tool being offered
                    return response.content, self._extract_winner_from_ruling(
                        response.content, winner
                    ), None

                convo.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))
                for tc in response.tool_calls:
                    if tc.name == "web_search" and searches_used < self.max_moderator_searches:
                        query = str(tc.arguments.get("query", "")).strip()
                        searches_used += 1
                        self._log(
                            f"  [Moderator searching ({searches_used}/{self.max_moderator_searches}): {query}]"
                        )
                        result = self.search_tool.search_formatted(query, max_results=3)
                    else:
                        result = "Search budget exhausted. Submit your verdict now."
                    convo.append(Message(role="tool", content=result, tool_call_id=tc.id))
        except TokenBudgetExceeded:
            raise  # a budget stop must not trigger the (token-spending) fallback
        except Exception as e:
            # Deliberately broad: any failure of the tool path degrades to the
            # text protocol (see turns.run_agent_turn).
            self._log(f"  [structured ruling failed ({e}); using text protocol]")
            logger.info("Structured ruling failed; using text protocol", exc_info=True)
            return None

        return None

    def _extract_winner_from_ruling(
        self, ruling: str, current_winner: Optional[str]
    ) -> Optional[str]:
        """Extract a winner from ruling text if one isn't already decided."""
        if current_winner:
            return current_winner
        ruling_upper = ruling.upper()
        if "FINAL VERDICT: PROSECUTION WINS" in ruling_upper or "PROSECUTION WINS" in ruling_upper:
            return "prosecution"
        if "FINAL VERDICT: DEFENSE WINS" in ruling_upper or "DEFENSE WINS" in ruling_upper:
            return "defense"
        if "FINAL VERDICT: DRAW" in ruling_upper:
            return "draw"
        return None

    def _process_ruling_searches(self, ruling: str, messages: list[Message]) -> str:
        """
        Process search requests in the moderator's ruling (text protocol).

        If the ruling contains [SEARCH: query] tags, perform the searches
        and ask the moderator to provide a final ruling with the results.
        """
        from ..tools.search import extract_search_queries

        queries = extract_search_queries(ruling)
        if not queries:
            return ruling

        # Limit searches
        queries = queries[:self.max_moderator_searches]

        # Perform searches
        search_results = []
        self._log(f"\n  [Moderator searching: {', '.join(queries)}]")
        for query in queries:
            results = self.search_tool.search_formatted(query, max_results=3)
            search_results.append(results)
            self._log(f"\n{results}")

        # Ask moderator to finalize ruling with search results
        search_context = "\n\n".join(search_results)
        followup = f"""
Based on your search results:

{search_context}

Now provide your FINAL ruling. Remember to include one of:
- "FINAL VERDICT: PROSECUTION WINS"
- "FINAL VERDICT: DEFENSE WINS"
- "FINAL VERDICT: DRAW"

Give your complete final ruling:
"""
        messages_with_search = messages + [
            Message(role="assistant", content=ruling),
            Message(role="user", content=followup),
        ]
        final_ruling = self.moderator.respond(messages_with_search)
        return final_ruling

    def _phase(self, stage: str, agent: Agent) -> None:
        """Announce a phase start and honor cancellation requests."""
        if self.should_stop and self.should_stop():
            raise SessionCancelled()
        if self.on_status:
            self.on_status(stage, agent)

    def _should_force_verdict(self) -> bool:
        """True when the remaining budget must be reserved for the ruling."""
        if not self.config.force_verdict:
            return False
        meter = getattr(self.moderator, "meter", None)
        if not meter or not getattr(meter, "hard_limit_tokens", None):
            return False
        tin, tout = meter.total_tokens()
        remaining = meter.hard_limit_tokens - (tin + tout)
        if remaining < FORCE_VERDICT_RESERVE_TOKENS:
            self._log(
                f"  [force verdict: {remaining:,} tokens left < "
                f"{FORCE_VERDICT_RESERVE_TOKENS:,} reserve — cutting to the ruling]"
            )
            return True
        return False

    def _check_concession(self, response: str) -> bool:
        """Check if the response contains a concession."""
        concession_patterns = [
            r"\bI\s+CONCEDE\b",
            r"\bI\s+CONCEDE\s+THE\s+POINT\b",
            r"\bI\s+CONCEDE\s+THIS\s+DEBATE\b",
        ]
        for pattern in concession_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

    def _record_turn(
        self,
        speaker: str,
        role: str,
        content: str,
        round_number: Optional[int] = None,
    ) -> None:
        """Record a turn in the transcript and notify callback."""
        self.transcript.add_entry(
            speaker=speaker,
            role=role,
            content=content,
            round_number=round_number,
        )

        if self.on_turn:
            self.on_turn(speaker, role, content)

        if self.config.verbose:
            self._log(f"\n[{speaker} ({role})]:\n{content}\n")

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is on."""
        if self.config.verbose:
            print(message)
