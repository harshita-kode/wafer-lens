import pytest

from src.llm.client import RetryingLLM


class FlakyLLM:
    """Fails `failures` times with the given error, then succeeds."""
    def __init__(self, failures, error="429 RESOURCE_EXHAUSTED"):
        self.failures, self.error, self.calls = failures, error, 0

    def generate(self, system, user):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.error)
        return "ok"


def no_sleep(_seconds):
    pass


def test_retry_succeeds_after_temporary_failures():
    inner = FlakyLLM(failures=2)
    assert RetryingLLM(inner, sleep=no_sleep).generate("s", "u") == "ok"
    assert inner.calls == 3


def test_permanent_error_is_not_retried():
    inner = FlakyLLM(failures=5, error="400 API key not valid")
    with pytest.raises(RuntimeError):
        RetryingLLM(inner, sleep=no_sleep).generate("s", "u")
    assert inner.calls == 1


def test_gives_up_after_max_retries():
    inner = FlakyLLM(failures=10)
    with pytest.raises(RuntimeError):
        RetryingLLM(inner, retries=3, sleep=no_sleep).generate("s", "u")
    assert inner.calls == 4


def test_backoff_doubles_each_time():
    waits = []
    RetryingLLM(FlakyLLM(failures=3), base_delay=2, sleep=waits.append).generate("s", "u")
    assert waits == [2, 4, 8]