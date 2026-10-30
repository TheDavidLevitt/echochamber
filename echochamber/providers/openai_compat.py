"""Shared implementation for OpenAI-compatible chat APIs.

OpenAI, Together.AI, and LM Studio all speak the same chat-completions
protocol; the concrete providers only differ in how the client is built.
"""

import json
import logging
import time
from typing import Callable, Optional

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDef
from .retry import call_with_retries, provider_error_types

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    """Base provider for any OpenAI-compatible chat-completions endpoint."""

    supports_tools = True

    def __init__(self, model: str, client):
        self.model = model
        self.client = client

    def _to_wire(
        self,
        messages: list[Message],
        system_prompt: Optional[str],
    ) -> list[dict]:
        wire = []
        if system_prompt:
            wire.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                wire.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            elif msg.role == "tool":
                wire.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
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
            "messages": self._to_wire(messages, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Stream text-only calls; tool calls arrive as deltas that would need
        # reassembly, so those requests stay non-streaming.
        if on_delta and not tools:
            try:
                return self._complete_streaming(kwargs, on_delta)
            except provider_error_types() as e:
                # Endpoint may not support streaming — fall through
                logger.info("Streaming request failed (%s); retrying without streaming", e)
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tool_choice:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        start = time.time()
        response = call_with_retries(
            lambda: self.client.chat.completions.create(**kwargs)
        )
        latency_ms = (time.time() - start) * 1000

        choice = response.choices[0]
        tool_calls = []
        for tc in choice.message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )

        return LLMResponse(
            content=choice.message.content or "",
            model=getattr(response, "model", None) or self.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
        )

    def _complete_streaming(self, kwargs: dict, on_delta) -> LLMResponse:
        """Stream a text-only completion, emitting fragments via on_delta."""
        stream_kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.time()
        stream = call_with_retries(
            lambda: self.client.chat.completions.create(**stream_kwargs)
        )
        parts: list[str] = []
        usage = None
        model = self.model
        for chunk in stream:
            if getattr(chunk, "model", None):
                model = chunk.model
            if chunk.usage:
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                fragment = chunk.choices[0].delta.content
                parts.append(fragment)
                on_delta(fragment)

        return LLMResponse(
            content="".join(parts),
            model=model,
            usage=usage,
            raw_response=None,
            latency_ms=(time.time() - start) * 1000,
        )
