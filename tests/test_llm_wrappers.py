import pytest

from src.llm.client import RetryingLLM
from src.llm.client import CachedLLM
from src.llm.client import RetryingLLM, TruncatedResponseError


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



class CountingLLM:
    def __init__(self, reply="answer"):
        self.reply, self.calls = reply, 0

    def generate(self, system, user):
        self.calls += 1
        return self.reply


def test_cache_returns_saved_answer(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path, model="m")
    assert llm.generate("s", "u") == "answer"
    assert llm.generate("s", "u") == "answer"
    assert inner.calls == 1 and llm.hits == 1


def test_different_prompt_is_a_cache_miss(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path, model="m")
    llm.generate("s", "u1")
    llm.generate("s", "u2")
    assert inner.calls == 2


def test_model_name_is_part_of_the_key(tmp_path):
    inner = CountingLLM()
    CachedLLM(inner, tmp_path, model="a").generate("s", "u")
    CachedLLM(inner, tmp_path, model="b").generate("s", "u")
    assert inner.calls == 2


def test_empty_answers_are_not_cached(tmp_path):
    inner = CountingLLM(reply="")
    llm = CachedLLM(inner, tmp_path, model="m")
    llm.generate("s", "u")
    llm.generate("s", "u")
    assert inner.calls == 2

class AlwaysTruncates:
    def __init__(self):
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        raise TruncatedResponseError("cut off")


def test_truncation_is_not_retried():
    inner = AlwaysTruncates()
    with pytest.raises(TruncatedResponseError):
        RetryingLLM(inner, sleep=no_sleep).generate("s", "u")
    assert inner.calls == 1


def test_truncated_answer_is_not_cached(tmp_path):
    inner = AlwaysTruncates()
    llm = CachedLLM(inner, tmp_path, model="m")
    for _ in range(2):
        with pytest.raises(TruncatedResponseError):
            llm.generate("s", "u")
    assert inner.calls == 2
    assert list(tmp_path.iterdir()) == []