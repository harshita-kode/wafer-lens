import json

import pytest

from src.config import CHROMA_DIR, LABEL_MAP_PATH

CLASSES = json.load(open(LABEL_MAP_PATH))

pytestmark = pytest.mark.skipif(
    not CHROMA_DIR.exists(), reason="Vector DB missing. Run: python -m src.rag.ingest"
)


@pytest.fixture(scope="module")
def retriever():
    from src.rag.retriever import KnowledgeRetriever
    return KnowledgeRetriever()


def make_prediction(label, confidence, uncertain, second="Edge-Loc"):
    return {
        "label": label, "confidence": confidence, "uncertain": uncertain,
        "top_k": [{"label": label, "confidence": confidence},
                  {"label": second, "confidence": 1 - confidence}],
    }


@pytest.mark.parametrize("pattern", CLASSES)
def test_returns_only_requested_pattern(retriever, pattern):
    chunks = retriever.retrieve(pattern, include_general=False)
    assert chunks
    assert all(c["pattern"] == pattern for c in chunks)


@pytest.mark.parametrize("pattern", CLASSES)
def test_required_sections_come_first(retriever, pattern):
    from src.rag.retriever import REQUIRED_SECTIONS
    sections = [c["section"] for c in retriever.retrieve(pattern, include_general=False)]
    assert sections[:2] == REQUIRED_SECTIONS


def test_general_chunk_is_appended(retriever):
    chunks = retriever.retrieve("Scratch")
    assert chunks[-1]["pattern"] == "general"


def test_sources_are_parsed_lists(retriever):
    for c in retriever.retrieve("Edge-Ring"):
        assert isinstance(c["sources"], list)
        assert all("title" in s for s in c["sources"])


def test_similarity_is_in_valid_range(retriever):
    for c in retriever.retrieve("Center"):
        assert -1.0 <= c["similarity"] <= 1.0


def test_unknown_pattern_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("edge ring")


def test_confident_prediction_uses_one_pattern(retriever):
    result = retriever.retrieve_for_prediction(make_prediction("Loc", 0.95, uncertain=False))
    assert result["patterns"] == ["Loc"]


def test_uncertain_prediction_uses_top_two(retriever):
    result = retriever.retrieve_for_prediction(make_prediction("Loc", 0.55, uncertain=True))
    assert result["patterns"] == ["Loc", "Edge-Loc"]
    found = {c["pattern"] for c in result["chunks"]}
    assert {"Loc", "Edge-Loc"} <= found


def test_no_duplicate_chunks(retriever):
    result = retriever.retrieve_for_prediction(make_prediction("Loc", 0.55, uncertain=True))
    ids = [c["id"] for c in result["chunks"]]
    assert len(ids) == len(set(ids))


def test_collect_sources_has_no_duplicates(retriever):
    from src.rag.retriever import collect_sources
    sources = collect_sources(retriever.retrieve("Scratch"))
    keys = [s.get("url") or s.get("title") for s in sources]
    assert len(keys) == len(set(keys))