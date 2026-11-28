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

SIMILAR = [{"label": "Scratch", "similarity": 0.97, "index": 1, "id": "train-1"},
           {"label": "Scratch", "similarity": 0.95, "index": 2, "id": "train-2"},
           {"label": "Loc", "similarity": 0.93, "index": 3, "id": "train-3"}]


def test_similar_wafers_section_counts_matches():
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(GOOD_REPLY), similar=SIMILAR)
    assert "2 of the 3 most similar past wafers were labelled Scratch" in report.markdown


def test_similar_section_comes_before_sources():
    md = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(GOOD_REPLY), similar=SIMILAR).markdown
    assert md.index("Similar past wafers") < md.index("**Sources:**")


def test_no_similar_section_when_none_given():
    md = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(GOOD_REPLY)).markdown
    assert "Similar past wafers" not in md

def test_body_has_only_the_written_sections():
    report = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(GOOD_REPLY), similar=SIMILAR)
    assert report.body.startswith("**What this usually means:**")
    assert "Pattern detected" not in report.body
    assert "Similar past wafers" not in report.body
    assert "**Sources:**" not in report.body

FACTS = {"dies": 100, "failed_dies": 30, "fail_rate": 0.3,
         "failure_share_by_zone": {"center": 0.1, "middle": 0.1, "edge": 0.8}}


def test_facts_section_comes_after_header():
    md = write_report(CONFIDENT, RETRIEVAL, llm=FakeLLM(GOOD_REPLY), facts=FACTS).markdown
    assert "**This wafer:** 30 of 100 dies failed (30%); 80% of failures are in the edge zone." in md
    assert md.index("Pattern detected") < md.index("This wafer") < md.index("What this usually means")


def test_use_llm_false_never_calls_llm():
    class ExplodingLLM:
        def generate(self, system, user):
            raise AssertionError("LLM must not be called")
    report = write_report(CONFIDENT, RETRIEVAL, llm=ExplodingLLM(), use_llm=False)
    assert report.generated_by == "template" and report.warnings == []


def test_feedback_is_added_to_prompt():
    fake = FakeLLM(GOOD_REPLY)
    write_report(CONFIDENT, RETRIEVAL, llm=fake, feedback=["2 bullet(s) have no citation"])
    assert "REVISION" in fake.last_user and "2 bullet(s) have no citation" in fake.last_user


def test_investigated_header_shows_cnn_opinion():
    decided = {"label": "Edge-Loc", "confidence": 0.72, "uncertain": False, "resolved_by": "investigation",
               "cnn_label": "Loc", "cnn_confidence": 0.55,
               "top_k": [{"label": "Edge-Loc", "confidence": 0.72}, {"label": "Loc", "confidence": 0.2}]}
    md = write_report(decided, RETRIEVAL, llm=FakeLLM(GOOD_REPLY)).markdown
    assert "Edge-Loc (evidence score 72%), resolved by investigation; the CNN alone said Loc (55%)" in md