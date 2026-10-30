"""Retry layer: typed-exception classification, Retry-After, full-jitter backoff."""

from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from echochamber.providers import retry
from echochamber.providers.retry import (
    backoff_delay,
    call_with_retries,
    is_retryable,
    retry_after_seconds,
)


# ---------------------------------------------------------------- fakes
# Shaped like the real SDK errors: anthropic/openai carry .status_code,
# google-genai carries .code; all expose .response.headers.

class FakeStatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


class FakeGenaiError(Exception):
    def __init__(self, code, headers=None):
        super().__init__(f"{code} error")
        self.code = code
        self.response = SimpleNamespace(headers=headers or {}) if headers is not None else None


class FakeConnectionError(Exception):
    pass


@pytest.fixture
def fake_sdks(monkeypatch):
    types = retry._ErrorTypes(
        status=((FakeStatusError, "status_code"), (FakeGenaiError, "code")),
        transient=(ConnectionError, TimeoutError, FakeConnectionError),
    )
    monkeypatch.setattr(retry, "_error_types", lambda: types)


class FixedRng:
    """uniform() that returns the upper bound, so delays are the jitter cap."""

    def __init__(self):
        self.calls = []

    def uniform(self, lo, hi):
        self.calls.append((lo, hi))
        return hi


# ---------------------------------------------------------- classification

@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
def test_transient_statuses_are_retryable(fake_sdks, status):
    assert is_retryable(FakeStatusError(status))
    assert is_retryable(FakeGenaiError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(fake_sdks, status):
    assert not is_retryable(FakeStatusError(status))
    assert not is_retryable(FakeGenaiError(status))


def test_connection_and_timeout_errors_are_retryable(fake_sdks):
    assert is_retryable(FakeConnectionError("Server disconnected"))
    assert is_retryable(ConnectionResetError("reset by peer"))
    assert is_retryable(TimeoutError("timed out"))


def test_unknown_exception_types_are_not_retried(fake_sdks):
    # The message mentions 503 and "connection", but the type is unknown:
    # message text no longer decides anything.
    assert not is_retryable(RuntimeError("503 connection reset"))
    assert not is_retryable(ValueError("rate limit"))


def test_status_error_without_a_status_is_not_retried(fake_sdks):
    err = FakeStatusError(503)
    err.status_code = None
    assert not is_retryable(err)


# --------------------------------------------------------------- Retry-After

def test_retry_after_seconds_header():
    assert retry_after_seconds(FakeStatusError(429, {"Retry-After": "7"})) == 7.0


def test_retry_after_ms_header_wins():
    err = FakeStatusError(429, {"retry-after-ms": "1500", "retry-after": "9"})
    assert retry_after_seconds(err) == 1.5


def test_retry_after_http_date():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    later = format_datetime(now + timedelta(seconds=30), usegmt=True)
    err = FakeStatusError(503, {"retry-after": later})
    assert retry_after_seconds(err, now=lambda: now.timestamp()) == pytest.approx(30.0)


def test_retry_after_in_the_past_is_zero():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    earlier = format_datetime(now - timedelta(seconds=30), usegmt=True)
    err = FakeStatusError(503, {"Retry-After": earlier})
    assert retry_after_seconds(err, now=lambda: now.timestamp()) == 0.0


def test_missing_or_garbage_retry_after():
    assert retry_after_seconds(FakeStatusError(503)) is None
    assert retry_after_seconds(FakeStatusError(503, {"retry-after": "soon"})) is None
    assert retry_after_seconds(FakeGenaiError(503)) is None  # no response object
    assert retry_after_seconds(RuntimeError("x")) is None


# ------------------------------------------------------------------ backoff

def test_full_jitter_cap_doubles_and_saturates():
    rng = FixedRng()
    delays = [backoff_delay(a, base_delay=1.0, max_delay=5.0, rng=rng) for a in range(5)]
    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0]
    assert all(lo == 0.0 for lo, _ in rng.calls)  # full jitter: lower bound is 0


def test_retries_then_succeeds_with_jittered_sleeps(fake_sdks):
    slept, rng = [], FixedRng()
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FakeStatusError(503)
        return "ok"

    assert call_with_retries(flaky, attempts=3, base_delay=1.0,
                             sleep=slept.append, rng=rng) == "ok"
    assert attempts["n"] == 3
    assert slept == [1.0, 2.0]


def test_retry_after_overrides_backoff_and_is_capped(fake_sdks):
    slept = []
    errors = iter([
        FakeStatusError(429, {"retry-after": "3"}),
        FakeStatusError(429, {"retry-after": "100000"}),
    ])

    def throttled():
        err = next(errors, None)
        if err:
            raise err
        return "ok"

    assert call_with_retries(throttled, attempts=3, sleep=slept.append, rng=FixedRng()) == "ok"
    assert slept == [3.0, retry.MAX_RETRY_AFTER_SECONDS]


def test_permanent_error_fails_fast(fake_sdks):
    slept, attempts = [], {"n": 0}

    def broken():
        attempts["n"] += 1
        raise FakeStatusError(401)

    with pytest.raises(FakeStatusError):
        call_with_retries(broken, attempts=3, sleep=slept.append)
    assert attempts["n"] == 1  # no pointless retries
    assert slept == []


def test_gives_up_after_last_attempt(fake_sdks):
    slept, attempts = [], {"n": 0}

    def down():
        attempts["n"] += 1
        raise FakeConnectionError("refused")

    with pytest.raises(FakeConnectionError):
        call_with_retries(down, attempts=3, sleep=slept.append, rng=FixedRng())
    assert attempts["n"] == 3
    assert len(slept) == 2


def test_on_retry_callback_sees_attempt_and_delay(fake_sdks):
    seen = []
    errors = iter([FakeStatusError(500)])

    def once():
        err = next(errors, None)
        if err:
            raise err
        return 1

    call_with_retries(once, on_retry=lambda n, d, e: seen.append((n, d, type(e))),
                      sleep=lambda s: None, rng=FixedRng())
    assert seen == [(1, 1.0, FakeStatusError)]


# ------------------------------------------- real SDK exception types (if installed)

def _httpx_response(status, headers=None):
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.example.invalid/v1")
    return httpx.Response(status, headers=headers or {}, request=request)


def test_real_anthropic_errors():
    anthropic = pytest.importorskip("anthropic")
    rate = anthropic.RateLimitError(
        "slow down", response=_httpx_response(429, {"retry-after": "4"}), body=None,
    )
    assert is_retryable(rate)
    assert retry_after_seconds(rate) == 4.0
    overloaded = anthropic.InternalServerError(
        "overloaded", response=_httpx_response(529), body=None,
    )
    assert is_retryable(overloaded)
    auth = anthropic.AuthenticationError("bad key", response=_httpx_response(401), body=None)
    assert not is_retryable(auth)
    request = _httpx_response(200).request
    assert is_retryable(anthropic.APIConnectionError(request=request))
    assert is_retryable(anthropic.APITimeoutError(request=request))


def test_real_openai_errors():
    openai = pytest.importorskip("openai")
    assert is_retryable(openai.InternalServerError(
        "boom", response=_httpx_response(500), body=None,
    ))
    assert not is_retryable(openai.BadRequestError(
        "bad", response=_httpx_response(400), body=None,
    ))
    request = _httpx_response(200).request
    assert is_retryable(openai.APIConnectionError(request=request))


def test_real_genai_and_transport_errors():
    errors = pytest.importorskip("google.genai.errors")
    httpx = pytest.importorskip("httpx")
    assert is_retryable(errors.ServerError(503, {"error": {"message": "unavailable"}}))
    assert is_retryable(errors.ClientError(429, {"error": {"message": "quota"}}))
    assert not is_retryable(errors.ClientError(403, {"error": {"message": "denied"}}))
    assert is_retryable(httpx.RemoteProtocolError(
        "Server disconnected without sending a response."
    ))
