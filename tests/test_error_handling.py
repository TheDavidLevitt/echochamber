"""Narrowed exception handlers: expected failures degrade and log, bugs propagate."""

import logging
from types import SimpleNamespace

import pytest

from echochamber.core.preprocessor import DocumentPreprocessor, PreprocessorConfig
from echochamber.providers import Message, retry
from echochamber.providers.openai_compat import OpenAICompatProvider


class FakeAPIError(Exception):
    """Stands in for an SDK status error (e.g. openai.BadRequestError)."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.fixture
def fake_sdk(monkeypatch):
    types = retry._ErrorTypes(status=((FakeAPIError, "status_code"),), transient=())
    monkeypatch.setattr(retry, "_error_types", lambda: types)


def _completion(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        model="fake",
    )


class CompatProvider(OpenAICompatProvider):
    @property
    def name(self) -> str:
        return "compat/m"


class StreamRejectingClient:
    """Rejects stream=True with `stream_error`; answers plain requests."""

    def __init__(self, stream_error):
        self.stream_error = stream_error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if kwargs.get("stream"):
            raise self.stream_error
        return _completion("plain answer")


def test_streaming_api_error_falls_back_and_logs(fake_sdk, caplog):
    provider = CompatProvider("m", StreamRejectingClient(FakeAPIError(400)))
    with caplog.at_level(logging.INFO, logger="echochamber.providers.openai_compat"):
        response = provider.complete([Message(role="user", content="hi")], on_delta=lambda s: None)
    assert response.content == "plain answer"
    assert "retrying without streaming" in caplog.text


def test_streaming_programming_error_is_not_swallowed(fake_sdk):
    provider = CompatProvider("m", StreamRejectingClient(KeyError("bug")))
    with pytest.raises(KeyError):
        provider.complete([Message(role="user", content="hi")], on_delta=lambda s: None)


def test_corrupt_cache_is_a_logged_miss(tmp_path, caplog):
    pre = DocumentPreprocessor(PreprocessorConfig(cache_dir=tmp_path))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "list.json").write_text("[1, 2]")
    with caplog.at_level(logging.WARNING, logger="echochamber.core.preprocessor"):
        assert pre._load_cache("bad") is None
        assert pre._load_cache("list") is None
    assert caplog.text.count("Ignoring unreadable cache file") == 2


def test_cache_round_trip_still_works(tmp_path):
    pre = DocumentPreprocessor(PreprocessorConfig(cache_dir=tmp_path))
    pre._save_cache("k", "summary text")
    assert pre._load_cache("k") == "summary text"
