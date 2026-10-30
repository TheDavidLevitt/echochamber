"""Anthropic Claude provider."""

import logging
import os
import time
from typing import Callable, Optional

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDef
from .retry import call_with_retries, provider_error_types

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""

    supports_tools = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        # Lazy import to avoid requiring anthropic if not used
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def _to_wire(self, messages: list[Message]) -> list[dict]:
        """Convert to Anthropic format.

        Assistant tool calls become tool_use content blocks; tool results
        become tool_result blocks in a user message. Consecutive tool results
        are merged into one user message, as the API requires all results for
        a turn's tool_use blocks to arrive together.
        """
        wire = []
        for msg in messages:
            if msg.role == "system":
                continue  # Anthropic handles system separately
            if msg.role == "assistant" and msg.tool_calls:
                blocks = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                wire.append({"role": "assistant", "content": blocks})
            elif msg.role == "tool":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(result_block)
                else:
                    wire.append({"role": "user", "content": [result_block]})
            else:
                wire.append({"role": msg.role, "content": msg.content})
        return wire

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[list[ToolDef]] = None,
        tool_choice: Optional[str] = None,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_wire(messages),
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if on_delta and not tools:
            try:
                return self._complete_streaming(kwargs, on_delta)
            except provider_error_types() as e:
                # Fall back to a normal (non-streaming) request
                logger.info("Streaming request failed (%s); retrying without streaming", e)
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
            if tool_choice:
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        start = time.time()
        response = call_with_retries(lambda: self.client.messages.create(**kwargs))
        latency_ms = (time.time() - start) * 1000

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            content="\n".join(text_parts),
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump(),
            tool_calls=tool_calls,
            latency_ms=latency_ms,
        )

    def _complete_streaming(self, kwargs: dict, on_delta) -> LLMResponse:
        """Stream a text-only completion, emitting fragments via on_delta."""
        start = time.time()
        with self.client.messages.stream(**kwargs) as stream:
            for fragment in stream.text_stream:
                on_delta(fragment)
            message = stream.get_final_message()

        text = "".join(b.text for b in message.content if b.type == "text")
        return LLMResponse(
            content=text,
            model=message.model,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
            raw_response=message.model_dump(),
            latency_ms=(time.time() - start) * 1000,
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"
