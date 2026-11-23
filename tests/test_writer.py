from src.report.prompts import label_sources, renumber_citations
from src.report.writer import write_report
from src.llm.client import TruncatedResponseError

SRC_A = {"id": 1, "title": "Paper A", "authors": "Author A", "year": 2024, "url": "https://a"}
SRC_B = {"id": 2, "title": "Paper B", "authors": "Author B", "year": 2025, "url": "https://b"}


def make_chunk(pattern, section, text, sources):
    return {"id": f"{pattern}::{section}", "pattern": pattern, "section": section,
            "title": f"{pattern} pattern", "text": f"{pattern} pattern — {section}\n\n{text}",
            "sources": sources, "similarity": 0.5}


CHUNKS = [
    make_chunk("Scratch", "Common causes", "- Handling damage [1]", [SRC_A]),
    make_chunk("Scratch", "What engineers usually check", "- Review handler paths [2]", [SRC_B]),
]
CONFIDENT = {"label": "Scratch", "confidence": 0.92, "uncertain": False,
             "top_k": [{"label": "Scratch", "confidence": 0.92}, {"label": "Loc", "confidence": 0.05}]}
UNCERTAIN = {"label": "Scratch", "confidence": 0.55, "uncertain": True,
             "top_k": [{"label": "Scratch", "confidence": 0.55}, {"label": "Loc", "confidence": 0.30}]}
RETRIEVAL = {"patterns": ["Scratch"], "chunks": CHUNKS}
GOOD_REPLY = ("**What this usually means:**\n- Often linked to handling damage [S1]\n\n"
              "**Suggested checks:**\n- Review handler paths [S2]")


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.last_user = reply, None

    def generate(self, system, user):
        self.last_user = user
        return self.reply


class BrokenLLM:
    def generate(self, system, user):
        raise RuntimeError("no API key")


def test_sources_get_global_labels():
    labels = label_sources(CHUNKS)
    assert [label for label, _ in labels.values()] == ["S1", "S2"]


def test_local_citations_are_renumbered():
    text = renumber_citations(CHUNKS[1], label_sources(CHUNKS))
    assert "[S2]" in text and "[2]" not in text


def test_header_uses_classifier_numbers():
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(GOOD_REPLY))
    assert "Scratch (92% confidence)" in report.markdown
    assert report.generated_by == "llm"
    assert report.warnings == []


def test_uncertain_header_mentions_second_pattern():
    report = write_report(UNCERTAIN, RETRIEVAL, llm=FakeLLM(GOOD_REPLY))
    assert "possibly Loc (30%)" in report.markdown


def test_only_cited_sources_are_listed():
    reply = "**What this usually means:**\n- Handling damage [S1]"
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(reply))
    assert [s["label"] for s in report.sources] == ["S1"]
    assert "Paper B" not in report.markdown


def test_invented_citation_is_removed():
    reply = "**What this usually means:**\n- Handling damage [S1][S9]"
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(reply))
    assert "[S9]" not in report.markdown
    assert any("S9" in w for w in report.warnings)


def test_missing_citations_are_flagged():
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM("**What this usually means:**\n- Something"))
    assert any("no citations" in w for w in report.warnings)


def test_falls_back_to_template_when_llm_fails():
    report = write_report(CONFIDENT, RETRIEVAL, llm=BrokenLLM())
    assert report.generated_by == "template"
    assert "Handling damage [S1]" in report.markdown
    assert any("template" in w for w in report.warnings)


def test_prompt_asks_for_both_patterns_when_uncertain():
    fake = FakeLLM(GOOD_REPLY)
    write_report(UNCERTAIN, {"patterns": ["Scratch", "Loc"], "chunks": CHUNKS}, llm=fake)
    assert "UNCERTAIN" in fake.last_user
    assert "How to tell them apart" in fake.last_user


def test_prompt_uses_only_global_labels():
    fake = FakeLLM(GOOD_REPLY)
    write_report(CONFIDENT, RETRIEVAL, llm=fake)
    assert "[1]" not in fake.last_user and "[S1]" in fake.last_user

from src.report.writer import normalize_bullets


def test_star_bullets_become_hyphens():
    text = "**What this usually means:**\n* First [S1]\n  * Nested [S2]"
    assert normalize_bullets(text) == "**What this usually means:**\n- First [S1]\n  - Nested [S2]"


def test_star_bullet_reply_is_normalized_in_report():
    reply = "**What this usually means:**\n* Handling damage [S1]"
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(reply))
    assert "- Handling damage [S1]" in report.markdown\


class TruncatingLLM:
    def generate(self, system, user):
        raise TruncatedResponseError("response cut off at 1500 output tokens")


def test_truncated_response_falls_back_to_template():
    report = write_report(CONFIDENT, RETRIEVAL, llm=TruncatingLLM())
    assert report.generated_by == "template"
    assert any("cut off" in w for w in report.warnings)