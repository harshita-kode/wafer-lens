import numpy as np
import pytest

from src.agent.graph import WaferAgent

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']


def make_pred(label, conf, second, second_conf):
    return {"label": label, "confidence": conf, "uncertain": conf < 0.7,
            "top_k": [{"label": label, "confidence": conf}, {"label": second, "confidence": second_conf}]}


class FakeClassifier:
    classes = CLASSES
    model_epoch = 26

    def __init__(self, prediction, tta=None):
        self.prediction = prediction
        self.tta = tta if tta is not None else np.full(len(CLASSES), 1 / len(CLASSES))

    def predict(self, wafer):
        return self.prediction

    def predict_proba_batch(self, wafers, batch_size=256):
        return np.tile(self.tta, (len(wafers), 1))

    def embed(self, wafer):
        return np.ones(128, dtype=np.float32)


class FakeIndex:
    def __init__(self, labels):
        self.labels = labels

    def find(self, embedding, k=5):
        return [{"id": f"train-{i}", "label": l, "index": i, "similarity": 0.95}
                for i, l in enumerate(self.labels[:k])]


def chunks_for(pattern):
    src = [{"id": 1, "title": f"Paper {pattern}", "authors": "A", "year": 2024, "url": f"https://{pattern}"}]
    return [{"id": f"{pattern}::{s}", "pattern": pattern, "section": s, "title": pattern,
             "text": f"{pattern} — {s}\n\n- A point [1]", "sources": src, "similarity": 0.5}
            for s in ("Common causes", "What engineers usually check", "Often confused with")]


class FakeRetriever:
    def retrieve_for_prediction(self, p):
        patterns = [t["label"] for t in p["top_k"][:2]] if p["uncertain"] else [p["label"]]
        return {"patterns": patterns, "chunks": [c for x in patterns for c in chunks_for(x)]}


class ScriptedLLM:
    """Returns a well-formed reply by default, or the given replies in order."""
    def __init__(self, replies=None):
        self.replies, self.prompts = replies, []

    def generate(self, system, user):
        self.prompts.append(user)
        if self.replies:
            return self.replies[min(len(self.prompts), len(self.replies)) - 1]
        sections = ["What this usually means", "Suggested checks"]
        if "UNCERTAIN" in user:
            sections.append("How to tell them apart")
        return "\n\n".join(f"**{s}:**\n- A point [S1]" for s in sections)


WAFER = np.ones((30, 30), dtype=np.uint8)


def make_agent(prediction, neighbours, tta=None, llm=None):
    return WaferAgent(classifier=FakeClassifier(prediction, tta), retriever=FakeRetriever(),
                      similar_index=FakeIndex(neighbours), llm=llm or ScriptedLLM())


def tta_for(**kw):
    p = np.zeros(len(CLASSES))
    for name, value in kw.items():
        p[CLASSES.index(name.replace("_", "-"))] = value
    return p


def nodes(result):
    return [s["node"] for s in result.agent_steps]


def test_confident_path_skips_investigation():
    result = make_agent(make_pred("Scratch", 0.95, "Loc", 0.03), ["Scratch"] * 5).analyze(WAFER)
    assert nodes(result) == ["classify", "measure", "similar", "decide", "retrieve", "write", "review"]
    assert result.investigation is None and result.generated_by == "llm"


def test_uncertain_wafer_is_resolved_when_evidence_agrees():
    agent = make_agent(make_pred("Loc", 0.55, "Edge-Loc", 0.30),
                       ["Edge-Loc"] * 4 + ["Loc"], tta=tta_for(Loc=0.3, Edge_Loc=0.6))
    result = agent.analyze(WAFER)
    assert "investigate" in nodes(result)
    assert result.prediction["label"] == "Edge-Loc" and not result.prediction["uncertain"]
    assert "resolved by investigation" in result.report_markdown


def test_conflicting_evidence_stays_uncertain():
    agent = make_agent(make_pred("Loc", 0.55, "Edge-Loc", 0.30),
                       ["Loc", "Loc", "Edge-Loc", "Edge-Loc", "Scratch"], tta=tta_for(Loc=0.45, Edge_Loc=0.40))
    result = agent.analyze(WAFER)
    assert result.prediction["uncertain"]
    assert set(result.patterns) == {"Loc", "Edge-Loc"}


def test_confident_none_skips_the_llm():
    llm = ScriptedLLM()
    result = make_agent(make_pred("none", 0.97, "Random", 0.01), ["none"] * 5, llm=llm).analyze(WAFER)
    assert "brief" in nodes(result) and llm.prompts == []
    assert result.generated_by == "template"


def test_bad_report_is_rewritten_once_with_feedback():
    bad = "**What this usually means:**\n- No citation here"
    good = "**What this usually means:**\n- A point [S1]\n\n**Suggested checks:**\n- A check [S1]"
    llm = ScriptedLLM(replies=[bad, good])
    result = make_agent(make_pred("Scratch", 0.95, "Loc", 0.03), ["Scratch"] * 5, llm=llm).analyze(WAFER)
    assert nodes(result).count("write") == 2
    assert "REVISION" in llm.prompts[1]
    assert "[S1]" in result.report_body and not any("still had issues" in w for w in result.warnings)


def test_rewrites_stop_at_the_limit():
    llm = ScriptedLLM(replies=["**What this usually means:**\n- No citation"])
    result = make_agent(make_pred("Scratch", 0.95, "Loc", 0.03), ["Scratch"] * 5, llm=llm).analyze(WAFER)
    assert nodes(result).count("write") == 2
    assert any("still had issues" in w for w in result.warnings)


def test_facts_and_timings_are_returned():
    result = make_agent(make_pred("Scratch", 0.95, "Loc", 0.03), ["Scratch"] * 5).analyze(WAFER)
    assert result.wafer_facts["dies"] == 900
    assert "total" in result.timings_ms and "This wafer" in result.report_markdown


def test_invalid_wafer_raises():
    with pytest.raises(ValueError):
        make_agent(make_pred("Scratch", 0.95, "Loc", 0.03), ["Scratch"] * 5).analyze(np.full((30, 30), 7))