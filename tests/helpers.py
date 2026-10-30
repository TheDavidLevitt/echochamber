"""Test doubles: a scripted LLM provider and a canned search tool."""

from echochamber.providers import LLMProvider, LLMResponse, ToolCall


class FakeProvider(LLMProvider):
    """Provider that replays scripted responses.

    Each item in responses is either a plain string (returned as text) or a
    prebuilt LLMResponse (for tool-call turns). Every call is recorded on
    .calls for assertions.
    """

    def __init__(self, responses, model="fake-model", supports_tools=False):
        self.model = model
        self.supports_tools = supports_tools
        self._responses = list(responses)
        self.calls = []

    def complete(
        self,
        messages,
        system_prompt=None,
        temperature=0.7,
        max_tokens=1024,
        tools=None,
        tool_choice=None,
        on_delta=None,
    ):
        self.calls.append({
            "messages": list(messages),
            "system_prompt": system_prompt,
            "tools": tools,
            "tool_choice": tool_choice,
            "on_delta": on_delta,
        })
        if not self._responses:
            raise AssertionError(f"{self.name} ran out of scripted responses")
        scripted = self._responses.pop(0)
        if isinstance(scripted, str):
            if on_delta and not tools:
                # Stream word-by-word like a real provider would
                words = scripted.split(" ")
                for i, word in enumerate(words):
                    on_delta(word + (" " if i < len(words) - 1 else ""))
            return LLMResponse(
                content=scripted,
                model=self.model,
                usage={"input_tokens": 100, "output_tokens": 50},
                latency_ms=1.0,
            )
        return scripted


    @property
    def name(self):
        return f"fake/{self.model}"


def tool_call_response(name, arguments, content="", call_id="call_1", model="fake-model"):
    """Build an LLMResponse containing a single tool call."""
    return LLMResponse(
        content=content,
        model=model,
        usage={"input_tokens": 100, "output_tokens": 50},
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        latency_ms=1.0,
    )


class FakeSearchTool:
    """Search tool returning canned results and recording queries."""

    def __init__(self):
        self.queries = []

    def search_formatted(self, query, max_results=3):
        self.queries.append(query)
        return f"=== Search Results for: {query} ===\n1. Canned result about {query}"
