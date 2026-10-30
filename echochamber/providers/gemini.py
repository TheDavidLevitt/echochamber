"""Google Gemini provider (google-genai SDK).

Reaches Gemini through either front door:

- Gemini API (AI Studio): set GEMINI_API_KEY (or GOOGLE_API_KEY).
- Vertex AI: set GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT
  (+ GOOGLE_APPLICATION_CREDENTIALS or gcloud ADC). Same models, but billed
  to the GCP project — where cloud credits apply.

Native tool calling is supported on both.
"""

import logging
import os
import time
from typing import Callable, Optional

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDef
from .retry import call_with_retries, provider_error_types

logger = logging.getLogger(__name__)

_SCHEMA_TYPE_MAP = {
    "object": "OBJECT", "string": "STRING", "number": "NUMBER",
    "integer": "INTEGER", "boolean": "BOOLEAN", "array": "ARRAY",
}


def _to_gemini_schema(schema: dict) -> dict:
    """Normalize a JSON Schema dict to Gemini's Schema shape (upper-case types)."""
    out = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = _SCHEMA_TYPE_MAP.get(value.lower(), value)
        elif key == "properties" and isinstance(value, dict):
            out[key] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _to_gemini_schema(value)
        else:
            out[key] = value
    return out


def build_client(api_key: Optional[str] = None):
    """Construct a google-genai Client.

    Returns:
        (client, mode) where mode is "vertex:<project>" or "gemini-api"
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")

    if use_vertex or (not api_key and project):
        if not project:
            raise ValueError(
                "Vertex mode needs GOOGLE_CLOUD_PROJECT (and ADC or GOOGLE_APPLICATION_CREDENTIALS)."
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        return genai.Client(vertexai=True, project=project, location=location), f"vertex:{project}"
    if api_key:
        return genai.Client(api_key=api_key), "gemini-api"
    raise ValueError(
        "Gemini needs GEMINI_API_KEY (Gemini API) or GOOGLE_GENAI_USE_VERTEXAI=true "
        "+ GOOGLE_CLOUD_PROJECT for Vertex AI."
    )


class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models via the google-genai SDK."""

    supports_tools = True

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.client, self.mode = build_client(api_key)

    def _to_contents(self, messages: list[Message]) -> list[dict]:
        contents = []
        for msg in messages:
            if msg.role == "system":
                continue  # handled via system_instruction
            if msg.role == "assistant" and msg.tool_calls:
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append({"function_call": {"name": tc.name, "args": tc.arguments}})
                contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool":
                # Gemini keys tool results by function NAME; our synthetic
                # ToolCall ids are "<name>::<index>" so the name is recoverable.
                name = (msg.tool_call_id or "tool").split("::")[0]
                contents.append({
                    "role": "user",
                    "parts": [{"function_response": {
                        "name": name,
                        "response": {"result": msg.content},
                    }}],
                })
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents

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
        config: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            config["system_instruction"] = system_prompt
        if tools:
            config["tools"] = [{
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": _to_gemini_schema(t.parameters),
                    }
                    for t in tools
                ],
            }]
            if tool_choice:
                config["tool_config"] = {"function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [tool_choice],
                }}

        contents = self._to_contents(messages) or [{"role": "user", "parts": [{"text": "Begin."}]}]

        if on_delta and not tools:
            try:
                return self._complete_streaming(contents, config, on_delta)
            except provider_error_types() as e:
                # Fall back to a normal (non-streaming) request
                logger.info("Streaming request failed (%s); retrying without streaming", e)

        start = time.time()
        response = call_with_retries(
            lambda: self.client.models.generate_content(
                model=self.model, contents=contents, config=config,
            )
        )
        latency_ms = (time.time() - start) * 1000

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidates = response.candidates or []
        parts = (candidates[0].content.parts or []) if candidates and candidates[0].content else []
        for i, part in enumerate(parts):
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append(ToolCall(
                    id=f"{fc.name}::{i}",
                    name=fc.name,
                    arguments=dict(fc.args or {}),
                ))
            elif getattr(part, "text", None):
                texts.append(part.text)

        usage = None
        meta = getattr(response, "usage_metadata", None)
        if meta:
            usage = {
                "input_tokens": meta.prompt_token_count or 0,
                "output_tokens": (meta.candidates_token_count or 0) + (getattr(meta, "thoughts_token_count", 0) or 0),
            }

        return LLMResponse(
            content="\n".join(texts),
            model=self.model,
            usage=usage,
            raw_response=None,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
        )

    def _complete_streaming(self, contents, config, on_delta) -> LLMResponse:
        """Stream a text-only completion, emitting fragments via on_delta."""
        start = time.time()
        stream = call_with_retries(
            lambda: self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config,
            )
        )
        parts: list[str] = []
        last_meta = None
        for chunk in stream:
            meta = getattr(chunk, "usage_metadata", None)
            if meta:
                last_meta = meta
            try:
                fragment = chunk.text
            except (AttributeError, ValueError):
                # Older SDKs raise instead of returning None for non-text chunks
                logger.debug("Skipping stream chunk without text", exc_info=True)
                fragment = None
            if fragment:
                parts.append(fragment)
                on_delta(fragment)

        usage = None
        if last_meta:
            usage = {
                "input_tokens": last_meta.prompt_token_count or 0,
                "output_tokens": (last_meta.candidates_token_count or 0)
                + (getattr(last_meta, "thoughts_token_count", 0) or 0),
            }
        return LLMResponse(
            content="".join(parts),
            model=self.model,
            usage=usage,
            raw_response=None,
            latency_ms=(time.time() - start) * 1000,
        )

    @property
    def name(self) -> str:
        return f"gemini/{self.model}"
