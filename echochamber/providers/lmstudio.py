"""LM Studio local provider (OpenAI-compatible API)."""

import logging
from typing import Optional

from .openai_compat import OpenAICompatProvider
from .retry import provider_error_types

logger = logging.getLogger(__name__)


def get_lmstudio_model(base_url: str = "http://localhost:1234/v1") -> Optional[str]:
    """
    Query LM Studio for the currently loaded model.

    Returns:
        Model name if available, None otherwise
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI(api_key="lm-studio", base_url=base_url)
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except provider_error_types() as e:
        # Not fatal: the provider keeps its placeholder model name and the
        # first real request reports the connection problem.
        logger.debug("LM Studio model auto-detect failed at %s: %s", base_url, e)
    return None


class LMStudioProvider(OpenAICompatProvider):
    """Provider for locally hosted LM Studio models.

    Tool support depends on the loaded model; the turn loop falls back to
    sentinel-tag parsing if a tool-enabled request fails.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        auto_detect_model: bool = True,
    ):
        self.base_url = base_url

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        # LM Studio doesn't need a real key
        client = OpenAI(api_key="lm-studio", base_url=base_url)

        if auto_detect_model and model == "local-model":
            detected = get_lmstudio_model(base_url)
            model = detected if detected else model

        super().__init__(model=model, client=client)

    @property
    def name(self) -> str:
        return f"lmstudio/{self.model}"
