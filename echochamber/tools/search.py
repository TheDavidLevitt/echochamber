"""Web search tool for agents.

Agents can search the web during their turns to find supporting evidence
or counter opposing arguments.
"""

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional

from ..providers.base import ToolDef

logger = logging.getLogger(__name__)

# Native tool definition, for providers that support tool calling.
# Providers that don't are driven via the [SEARCH: query] sentinel below.
WEB_SEARCH_TOOL = ToolDef(
    name="web_search",
    description=(
        "Search the web for evidence, data, precedents, or expert opinions. "
        "Returns titles, snippets, and source URLs. You have a limited search "
        "budget per turn, so choose impactful queries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
        },
        "required": ["query"],
    },
)


def normalize_query(query: str) -> str:
    """
    Normalize a search query for better compatibility.

    - Converts accented characters to ASCII equivalents
    - Removes problematic characters
    - Handles French/European text better
    """
    # Normalize unicode (decompose accented characters)
    normalized = unicodedata.normalize('NFKD', query)
    # Remove combining characters (accents) but keep base letters
    ascii_text = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Clean up extra whitespace
    return ' '.join(ascii_text.split())


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str

    def __str__(self) -> str:
        return f"**{self.title}**\n{self.snippet}\nSource: {self.url}"


class WebSearchTool:
    """
    Web search tool using DuckDuckGo.

    No API key required - uses DuckDuckGo's free search.
    """

    def __init__(self, max_results: int = 5, retry_attempts: int = 2):
        """
        Initialize the search tool.

        Args:
            max_results: Maximum results to return per search
            retry_attempts: Number of retry attempts on failure
        """
        self.max_results = max_results
        self.retry_attempts = retry_attempts
        self._ddg = None

    def _get_ddg(self):
        """Lazy load DuckDuckGo search."""
        if self._ddg is None:
            try:
                # Try new ddgs package first
                from ddgs import DDGS
                self._ddg = DDGS()
            except ImportError:
                try:
                    # Fall back to old duckduckgo_search package
                    from duckduckgo_search import DDGS
                    self._ddg = DDGS()
                except ImportError:
                    raise ImportError(
                        "Web search requires ddgs. "
                        "Install with: pip install ddgs"
                    )
        return self._ddg

    def _reset_ddg(self):
        """Reset the DuckDuckGo client (useful after errors)."""
        self._ddg = None

    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        """
        Search the web for a query.

        Args:
            query: Search query
            max_results: Override default max results

        Returns:
            List of SearchResult objects
        """
        n = max_results or self.max_results

        # Try with original query first, then normalized
        queries_to_try = [query]
        normalized = normalize_query(query)
        if normalized != query:
            queries_to_try.append(normalized)

        for attempt_query in queries_to_try:
            for attempt in range(self.retry_attempts):
                try:
                    ddg = self._get_ddg()
                    results = list(ddg.text(attempt_query, max_results=n))
                    if results:
                        return [
                            SearchResult(
                                title=r.get("title", ""),
                                url=r.get("href", r.get("link", "")),
                                snippet=r.get("body", r.get("snippet", "")),
                            )
                            for r in results
                        ]
                except Exception as e:
                    # Broad on purpose: ddgs raises its own and network errors,
                    # and a failed search must never end the debate.
                    logger.debug("Search attempt failed for %r", attempt_query, exc_info=True)
                    error_str = str(e).lower()
                    # Reset client on connection errors
                    if "connection" in error_str or "broken pipe" in error_str:
                        self._reset_ddg()

                    if attempt < self.retry_attempts - 1:
                        time.sleep(0.5 * (attempt + 1))  # Brief delay before retry
                    else:
                        print(f"Search error for '{attempt_query}': {e}")

        return []

    def search_formatted(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Search and return formatted results as a string.

        Args:
            query: Search query
            max_results: Override default max results

        Returns:
            Formatted string with search results
        """
        results = self.search(query, max_results)

        if not results:
            return f"No results found for: {query}"

        lines = [f"=== Search Results for: {query} ===\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r}")
            lines.append("")

        return "\n".join(lines)


# Pattern to detect search requests in agent output
SEARCH_PATTERN = re.compile(
    r'\[SEARCH:\s*([^\]]+)\]',
    re.IGNORECASE
)


def extract_search_queries(text: str) -> list[str]:
    """
    Extract search queries from agent output.

    Agents can request searches by including [SEARCH: query] in their response.

    Args:
        text: Agent's response text

    Returns:
        List of search queries found
    """
    matches = SEARCH_PATTERN.findall(text)
    return [m.strip() for m in matches if m.strip()]


def process_searches(text: str, search_tool: WebSearchTool) -> tuple[str, str]:
    """
    Process any search requests in agent output.

    Args:
        text: Agent's response text
        search_tool: WebSearchTool instance

    Returns:
        Tuple of (cleaned_text, search_results)
        - cleaned_text: Original text with [SEARCH:] tags removed
        - search_results: Formatted search results (empty if no searches)
    """
    queries = extract_search_queries(text)

    if not queries:
        return text, ""

    # Remove search tags from text
    cleaned = SEARCH_PATTERN.sub('', text).strip()

    # Perform searches
    results_parts = []
    for query in queries:
        results_parts.append(search_tool.search_formatted(query))

    return cleaned, "\n\n".join(results_parts)
