"""Agent turn execution — one place that knows how to run a turn with tools.

Two transports for the same behavior:

- Native: providers with tool support get a web_search tool; the loop
  executes each requested search and feeds results back until the agent
  produces its final text.
- Sentinel: other providers request searches with [SEARCH: query] tags;
  the loop extracts them, runs them, and re-prompts once with the results.

Either way, search results reach the agent within the same turn.
"""

import logging
from typing import Callable, Optional

from ..providers import Message
from .agent import Agent
from .usage import TokenBudgetExceeded

logger = logging.getLogger(__name__)

# Iterations beyond the search budget, to give the agent a closing call
# after its last results arrive.
_LOOP_SLACK = 2


def run_agent_turn(
    agent: Agent,
    messages: list[Message],
    search_tool: Optional[object] = None,
    max_searches: int = 0,
    log: Callable[[str], None] = lambda s: None,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run one agent turn, executing any web searches the agent requests.

    Args:
        agent: The agent taking the turn
        messages: Conversation history plus this turn's instruction
        search_tool: WebSearchTool, or None to disable search
        max_searches: Search budget for this turn (0 = disabled)
        log: Progress logger
        on_delta: Streams text fragments as they generate (text-only calls;
            tool-using calls arrive whole)

    Returns:
        The agent's final text for the turn
    """
    search_enabled = search_tool is not None and max_searches > 0

    if search_enabled and agent.provider.supports_tools:
        try:
            return _native_turn(agent, messages, search_tool, max_searches, log)
        except TokenBudgetExceeded:
            raise  # a budget stop must not trigger the (token-spending) fallback
        except Exception as e:
            # Deliberately broad: providers are pluggable and some endpoints
            # (notably local models via LM Studio) reject tool-enabled
            # requests in their own ways; degrade to the sentinel transport.
            log(f"  [{agent.name}: native tools failed ({e}); falling back to text protocol]")
            logger.info(
                "%s: native tools failed; falling back to text protocol",
                agent.name, exc_info=True,
            )

    text = agent.respond_full(messages, on_delta=on_delta).content
    if not search_enabled:
        return text
    return _sentinel_followup(agent, messages, text, search_tool, max_searches, log)


def _native_turn(
    agent: Agent,
    messages: list[Message],
    search_tool,
    max_searches: int,
    log: Callable[[str], None],
) -> str:
    from ..tools.search import WEB_SEARCH_TOOL

    convo = list(messages)
    searches_used = 0
    response = None

    for _ in range(max_searches + _LOOP_SLACK):
        tools = [WEB_SEARCH_TOOL] if searches_used < max_searches else None
        response = agent.respond_full(convo, tools=tools)

        if not response.tool_calls:
            return response.content

        convo.append(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        ))
        for tc in response.tool_calls:
            if tc.name == "web_search" and searches_used < max_searches:
                query = str(tc.arguments.get("query", "")).strip()
                searches_used += 1
                log(f"  [{agent.name} searching ({searches_used}/{max_searches}): {query}]")
                result = search_tool.search_formatted(query, max_results=3)
            else:
                result = (
                    "Search budget for this turn is exhausted. "
                    "Give your final statement using what you already have."
                )
            convo.append(Message(role="tool", content=result, tool_call_id=tc.id))

    # Budget and slack exhausted with the agent still requesting tools;
    # take whatever text accompanied the last response.
    return response.content or "(no final statement produced)"


def _sentinel_followup(
    agent: Agent,
    messages: list[Message],
    text: str,
    search_tool,
    max_searches: int,
    log: Callable[[str], None],
) -> str:
    from ..tools.search import SEARCH_PATTERN, extract_search_queries

    queries = extract_search_queries(text)[:max_searches]
    if not queries:
        return text

    results = []
    for i, query in enumerate(queries, 1):
        log(f"  [{agent.name} searching ({i}/{len(queries)}): {query}]")
        results.append(search_tool.search_formatted(query, max_results=3))

    followup = (
        "Here are the results of your searches:\n\n"
        + "\n\n".join(results)
        + "\n\nIncorporate anything useful and restate your complete statement "
        "for this turn. Do not request further searches."
    )
    convo = messages + [
        Message(role="assistant", content=text),
        Message(role="user", content=followup),
    ]
    final = agent.respond(convo)
    # Strip any stray search tags so they don't leak into the transcript
    return SEARCH_PATTERN.sub("", final).strip()
