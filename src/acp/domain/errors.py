"""Failure classification.

PURE MODULE -- builtins only, no I/O, no imports from acp.db / acp.api.
Enforced by tests/unit/test_purity.py.

WHY CLASSIFY AT ALL
-------------------
"Did it fail?" is not a useful question. "Will it succeed if we try again?"
is, and the answer differs completely by cause:

  a 429 from an LLM provider    retry, but SLOWER -- retrying fast makes the
                                rate limit worse, not better
  a malformed payload           never retry; it is guaranteed to fail again,
                                and three attempts just triple the log noise
  the worker was killed         retry IMMEDIATELY -- the task never got a
                                chance to fail, so there is nothing to back
                                off from
  a connection refused          retry with a long cap; the dependency is down
                                and hammering it delays its recovery

A two-class system (retryable / not) collapses all of that into one backoff
curve, which is wrong for at least three of the four.

WHERE THE CLASS COMES FROM
--------------------------
Preferably from the adapter, which knows what it called and what came back.
`classify()` is the fallback for exceptions that arrive untyped -- it maps a
few well-known builtins and gives up honestly with UNKNOWN rather than
guessing PERMANENT (which would silently drop recoverable work) or guessing
TRANSIENT (which would retry real bugs three times).
"""

from __future__ import annotations

from enum import StrEnum


class FailureClass(StrEnum):
    """Why an attempt failed, in terms of what to do about it.

    This is a CLOSED set, and deliberately small. It is stored in
    `tasks.error_class`, used as a Prometheus label, and queried when
    debugging -- all three of which want a bounded vocabulary. New causes
    should map onto an existing class unless they genuinely need different
    retry behaviour.
    """

    #: Generic recoverable failure. The default for an adapter that knows the
    #: failure is worth retrying but has nothing more specific to say.
    TRANSIENT = "TRANSIENT"

    #: A quota or throttle. Retried more slowly than TRANSIENT, and honours a
    #: server-supplied Retry-After when one is available.
    RATE_LIMITED = "RATE_LIMITED"

    #: The call did not answer in time. Retryable, but note the side effect
    #: may still have landed -- a timeout says nothing about whether the
    #: request was processed, which is why step-level idempotency matters.
    TIMEOUT = "TIMEOUT"

    #: A downstream service is unreachable or erroring. Long cap: hammering a
    #: struggling dependency delays its recovery.
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"

    #: The attempt was taken away, not failed -- the worker died, was paused,
    #: or lost its lease. Retried with NO backoff, because the task itself
    #: never misbehaved. Written by the reaper, never raised by an adapter.
    WORKER_LOST = "WORKER_LOST"

    #: Will never succeed. Retrying is pure waste.
    PERMANENT = "PERMANENT"

    #: The caller's input is wrong. Also never retried, but distinguished from
    #: PERMANENT because it is the tenant's bug, not ours -- and that
    #: distinction decides who gets paged.
    USER_ERROR = "USER_ERROR"

    #: Policy refused the operation (agent/tool authorization, Phase 7).
    #: Never retried: the answer will not change, and retrying would turn one
    #: denial into max_attempts audit records.
    PERMISSION_DENIED = "PERMISSION_DENIED"

    #: We could not tell. Retried conservatively -- see retry.POLICIES, which
    #: caps UNKNOWN at fewer attempts than a known-transient failure, because
    #: an unclassified error is as likely to be a bug as a blip.
    UNKNOWN = "UNKNOWN"


#: Classes that are never worth another attempt.
TERMINAL_CLASSES: frozenset[FailureClass] = frozenset(
    {FailureClass.PERMANENT, FailureClass.USER_ERROR, FailureClass.PERMISSION_DENIED}
)


class AdapterError(Exception):
    """Base for errors an adapter raises to state its own failure class.

    Adapters are encouraged to raise these rather than bare exceptions: the
    adapter knows whether the API it called said "slow down" or "that input is
    invalid", and no amount of inspection at the worker can recover that.
    """

    failure_class: FailureClass = FailureClass.UNKNOWN

    def __init__(self, message: str = "", *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        #: Server-supplied delay (an HTTP Retry-After, say). When present the
        #: retry policy will not schedule sooner than this, whatever its own
        #: backoff curve says -- the server knows more than we do.
        self.retry_after_s = retry_after_s


class Retryable(AdapterError):
    """Generic recoverable failure."""

    failure_class = FailureClass.TRANSIENT


class RateLimited(AdapterError):
    failure_class = FailureClass.RATE_LIMITED


class UpstreamTimeout(AdapterError):
    failure_class = FailureClass.TIMEOUT


class DependencyUnavailable(AdapterError):
    failure_class = FailureClass.DEPENDENCY_UNAVAILABLE


class PermanentFailure(AdapterError):
    failure_class = FailureClass.PERMANENT


class InvalidInput(AdapterError):
    failure_class = FailureClass.USER_ERROR


class PermissionDenied(AdapterError):
    failure_class = FailureClass.PERMISSION_DENIED


def classify(exc: BaseException) -> FailureClass:
    """Best-effort classification of an exception the adapter did not type.

    Only well-known builtins are mapped. Everything else is UNKNOWN, on
    purpose: guessing PERMANENT would silently discard recoverable work, and
    guessing TRANSIENT would retry genuine bugs to exhaustion. UNKNOWN is the
    honest answer, and its retry policy is tuned for exactly that uncertainty.
    """
    if isinstance(exc, AdapterError):
        return exc.failure_class
    # asyncio.TimeoutError is an alias of the builtin from 3.11 onward.
    if isinstance(exc, TimeoutError):
        return FailureClass.TIMEOUT
    if isinstance(exc, ConnectionError):
        return FailureClass.DEPENDENCY_UNAVAILABLE
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        # A shape error on the payload. Almost always the submitter's bug, and
        # retrying identical input cannot fix it.
        return FailureClass.USER_ERROR
    if isinstance(exc, OSError):
        return FailureClass.DEPENDENCY_UNAVAILABLE
    return FailureClass.UNKNOWN


def retry_after_of(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after_s", None)
    return value if isinstance(value, (int, float)) and value >= 0 else None
