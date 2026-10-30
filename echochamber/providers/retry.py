"""Centralized retry with exponential backoff for provider API calls.

What gets retried is decided from each SDK's typed exceptions, not from
the error message text:

- HTTP status errors (``anthropic.APIStatusError``, ``openai.APIStatusError``,
  ``google.genai.errors.APIError``) are retried for 408, 409, 429 and 5xx.
  Everything else (400 bad request, 401/403 auth, 404 unknown model, ...)
  fails immediately, because repeating the same request cannot fix it.
- Connection and timeout errors (the SDKs' ``APIConnectionError`` /
  ``APITimeoutError``, the httpx/requests transport errors google-genai
  surfaces directly, and the builtin ``ConnectionError``/``TimeoutError``)
  are always retried.
- Any other exception type is not retried.

Waits use exponential backoff with full jitter, unless the server sent a
``Retry-After`` header, which takes precedence.

SDKs are imported lazily and only if installed, so the core package keeps
its zero-hard-dependency promise.
"""

import logging
import random
import time
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 408 Request Timeout, 409 Conflict (Anthropic/OpenAI use it for transient
# lock contention), 429 Too Many Requests. All 5xx are retried as well,
# including Anthropic's 529 "overloaded".
RETRYABLE_STATUS = frozenset({408, 409, 429})

# Never sleep longer than this on a server's say-so; a Retry-After of an
# hour is better surfaced as an error than silently waited out.
MAX_RETRY_AFTER_SECONDS = 60.0


class _ErrorTypes(NamedTuple):
    # (exception type, name of the attribute holding its HTTP status code)
    status: tuple
    # exception types that mean the request never got a usable response
    transient: tuple


@lru_cache(maxsize=1)
def _error_types() -> _ErrorTypes:
    """Exception types from whichever SDKs are installed."""
    status: list = []
    transient: list = [ConnectionError, TimeoutError]
    try:
        import anthropic
        status.append((anthropic.APIStatusError, "status_code"))
        transient.append(anthropic.APIConnectionError)  # includes APITimeoutError
    except ImportError:
        pass
    try:
        import openai
        status.append((openai.APIStatusError, "status_code"))
        transient.append(openai.APIConnectionError)  # includes APITimeoutError
    except ImportError:
        pass
    try:
        from google.genai import errors as genai_errors
        status.append((genai_errors.APIError, "code"))
    except ImportError:
        pass
    try:
        import httpx
        # google-genai lets transport errors through unwrapped, e.g. the
        # RemoteProtocolError "Server disconnected without sending a response".
        transient.append(httpx.TransportError)
    except ImportError:
        pass
    try:
        import requests
        transient.extend([requests.ConnectionError, requests.Timeout])
    except ImportError:
        pass
    return _ErrorTypes(status=tuple(status), transient=tuple(transient))


def provider_error_types() -> tuple:
    """Every API and transport error type the installed SDKs can raise.

    For handlers that should catch "the provider call failed" and nothing
    else (programming errors still propagate).
    """
    types = _error_types()
    return tuple(t for t, _ in types.status) + types.transient


def status_code(exc: BaseException) -> Optional[int]:
    """HTTP status carried by a typed SDK error, or None."""
    for exc_type, attr in _error_types().status:
        if isinstance(exc, exc_type):
            value = getattr(exc, attr, None)
            return value if isinstance(value, int) else None
    return None


def is_retryable(exc: BaseException) -> bool:
    """Whether repeating the request could plausibly succeed."""
    types = _error_types()
    if any(isinstance(exc, t) for t, _ in types.status):
        code = status_code(exc)
        return code is not None and (code in RETRYABLE_STATUS or 500 <= code <= 599)
    return isinstance(exc, types.transient)


def _header(headers, name: str) -> Optional[str]:
    """Case-insensitive header lookup over httpx/requests/plain-dict headers."""
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except AttributeError:
        return None
    if value is not None:
        return value
    lowered = name.lower()
    try:
        items = headers.items()
    except AttributeError:
        return None
    for key, val in items:
        if str(key).lower() == lowered:
            return val
    return None


def retry_after_seconds(
    exc: BaseException, now: Callable[[], float] = time.time
) -> Optional[float]:
    """Server-requested wait from Retry-After (seconds or HTTP-date), if any.

    Also honors ``retry-after-ms``, which Anthropic and OpenAI send.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)

    millis = _header(headers, "retry-after-ms")
    if millis is not None:
        try:
            return max(0.0, float(millis) / 1000.0)
        except ValueError:
            pass

    value = _header(headers, "retry-after")
    if value is None:
        return None
    value = str(value).strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - now())


def backoff_delay(
    attempt: int, base_delay: float, max_delay: float, rng=random
) -> float:
    """Exponential backoff with full jitter.

    attempt 0 waits up to base_delay, attempt 1 up to 2x, attempt 2 up to
    4x, ... capped at max_delay; the actual wait is uniform in [0, cap].
    """
    return rng.uniform(0.0, min(max_delay, base_delay * (2 ** attempt)))


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    rng=random,
) -> T:
    """
    Call fn, retrying transient failures.

    Args:
        fn: Zero-argument callable performing the API request
        attempts: Total attempts including the first
        base_delay: Backoff scale; the cap doubles with each retry
        max_delay: Upper bound on the backoff cap
        on_retry: Optional callback(attempt_number, delay_seconds, exception)
        sleep: Injectable sleep function (tests pass a recorder)
        rng: Injectable source of uniform() for the jitter

    Returns:
        fn's return value

    Raises:
        The last exception if all attempts fail, or immediately for
        non-retryable errors (auth, bad request, unknown exception types).
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not is_retryable(e):
                raise
            server_wait = retry_after_seconds(e)
            if server_wait is not None:
                delay = min(server_wait, MAX_RETRY_AFTER_SECONDS)
            else:
                delay = backoff_delay(attempt, base_delay, max_delay, rng)
            logger.warning(
                "Transient provider error (%s: %s); retry %d/%d in %.1fs",
                type(e).__name__, e, attempt + 1, attempts - 1, delay,
            )
            if on_retry:
                on_retry(attempt + 1, delay, e)
            sleep(delay)
    raise RuntimeError("unreachable")  # for type-checkers
