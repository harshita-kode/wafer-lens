import json

import pytest

from src.classifier.preprocess import validate_wafer
from src.pipeline import WaferAnalysisPipeline

PREDICTION = {"label": "Scratch", "confidence": 0.92, "uncertain": False,
              "top_k": [{"label": "Scratch", "confidence": 0.92}, {"label": "Loc", "confidence": 0.05}]}
CHUNKS = [{"id": "scratch::common-causes", "pattern": "Scratch", "section": "Common causes",
           "title": "Scratch defect pattern", "text": "Scratch — Common causes\n\n- Handling damage [1]",
           "sources": [{"id": 1, "title": "Paper A", "authors": "A", "year": 2024, "url": "https://a"}],
           "similarity": 0.6}]
NEIGHBOURS = [{"id": "train-1", "label": "Scratch", "index": 1, "similarity": 0.97},
              {"id": "train-2", "label": "Loc", "index": 2, "similarity": 0.95}]


class FakeClassifier:
    def predict(self, wafer_map):
        validate_wafer(wafer_map)
        return PREDICTION

    def embed(self, wafer_map):
        return [0.1] * 128


class FakeRetriever:
    def retrieve_for_prediction(self, prediction):
        return {"patterns": [prediction["label"]], "chunks": CHUNKS}


class FakeIndex:
    def find(self, embedding, k=5):
        return NEIGHBOURS[:k]


class FakeLLM:
    def generate(self, system, user):
        return "**What this usually means:**\n- Handling damage [S1]\n\n**Suggested checks:**\n- Check [S1]"


WAFER = [[1] * 20 for _ in range(20)]


def make_pipeline(**overrides):
    parts = dict(classifier=FakeClassifier(), retriever=FakeRetriever(),
                 similar_index=FakeIndex(), llm=FakeLLM())
    parts.update(overrides)
    return WaferAnalysisPipeline(**parts)


def test_full_result_has_every_part():
    result = make_pipeline().analyze(WAFER)
    assert result.prediction["label"] == "Scratch"
    assert result.generated_by == "llm"
    assert len(result.similar_wafers) == 2
    assert "Similar past wafers" in result.report_markdown
    assert result.sources[0]["title"] == "Paper A"


def test_timings_cover_each_step():
    timings = make_pipeline().analyze(WAFER).timings_ms
    assert set(timings) == {"classify", "similar", "retrieve", "report", "total"}


def test_works_without_similar_index():
    result = make_pipeline(use_similar=False, similar_index=None).analyze(WAFER)
    assert result.similar_wafers == []
    assert any("Similar-wafer" in w for w in result.warnings)


def test_invalid_wafer_raises_value_error():
    with pytest.raises(ValueError):
        make_pipeline().analyze([[7] * 20 for _ in range(20)])


def test_result_is_json_serialisable():
    json.dumps(make_pipeline().analyze(WAFER).to_dict())