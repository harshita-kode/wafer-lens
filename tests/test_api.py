import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.classifier.preprocess import validate_wafer
from src.api.schemas import AnalyzeRequest
from src.pipeline import AnalysisResult

PREDICTION = {"label": "Scratch", "confidence": 0.92, "uncertain": False,
              "top_k": [{"label": "Scratch", "confidence": 0.92}, {"label": "Loc", "confidence": 0.05}]}


class FakePipeline:
    def __init__(self, llm=True):
        self.classifier = type("C", (), {"model_epoch": 26})()
        self.retriever = object()
        self.similar_index = object()
        self.llm = object() if llm else None

    def analyze(self, wafer_map):
        validate_wafer(wafer_map)
        return AnalysisResult(
            prediction=PREDICTION,
            patterns=["Scratch"],
            report_markdown="**Pattern detected:** Scratch (92% confidence)",
            sources=[{"label": "S1", "id": 1, "title": "Paper A", "authors": "A",
                      "year": 2024, "url": "https://a"}],
            similar_wafers=[{"id": "train-1", "label": "Scratch", "index": 1, "similarity": 0.97}],
            generated_by="llm",
            timings_ms={"total": 12.0},
        )


@pytest.fixture
def samples_dir(tmp_path):
    np.save(tmp_path / "scratch.npy", np.ones((20, 20), dtype=np.uint8))
    return tmp_path


@pytest.fixture
def client(samples_dir):
    app = create_app(pipeline_factory=FakePipeline, samples_dir=samples_dir)
    with TestClient(app) as c:          # 'with' runs the lifespan (startup/shutdown)
        yield c


def npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


WAFER = [[1] * 20 for _ in range(20)]


# ---------- health ----------

def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_epoch"] == 26


def test_health_degraded_without_llm(samples_dir):
    app = create_app(pipeline_factory=lambda: FakePipeline(llm=False), samples_dir=samples_dir)
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"


# ---------- /analyze (JSON) ----------

def test_analyze_json_returns_full_result(client):
    response = client.post("/analyze", json={"wafer_map": WAFER})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["label"] == "Scratch"
    assert body["sources"][0] == {"label": "S1", "title": "Paper A", "authors": "A",
                                  "year": 2024, "url": "https://a"}   # internal 'id' dropped


def test_invalid_values_return_422_with_message(client):
    response = client.post("/analyze", json={"wafer_map": [[7] * 20] * 20})
    assert response.status_code == 422
    assert "0, 1, 2" in response.json()["detail"]


def test_ragged_rows_return_422(client):
    response = client.post("/analyze", json={"wafer_map": [[1] * 20, [1] * 5] * 10})
    assert response.status_code == 422


def test_oversized_wafer_returns_422(client):
    response = client.post("/analyze", json={"wafer_map": [[1] * 600] * 20})
    assert response.status_code == 422
    assert "maximum" in response.json()["detail"]


def test_missing_field_returns_422(client):
    assert client.post("/analyze", json={"map": WAFER}).status_code == 422


# ---------- /analyze/file ----------

def test_analyze_file_upload(client):
    files = {"file": ("wafer.npy", npy_bytes(np.ones((20, 20), dtype=np.uint8)))}
    assert client.post("/analyze/file", files=files).status_code == 200


def test_wrong_extension_returns_400(client):
    files = {"file": ("wafer.txt", b"not a wafer")}
    assert client.post("/analyze/file", files=files).status_code == 400


def test_pickled_upload_is_rejected(client):
    malicious = npy_bytes(np.array([{"a": 1}], dtype=object))     # needs pickle to load
    files = {"file": ("wafer.npy", malicious)}
    assert client.post("/analyze/file", files=files).status_code == 400


# ---------- samples ----------

def test_list_samples(client):
    assert client.get("/samples").json() == {"samples": ["scratch"]}


def test_get_sample(client):
    body = client.get("/samples/scratch").json()
    assert body["name"] == "scratch" and len(body["wafer_map"]) == 20


def test_unknown_sample_returns_404(client):
    assert client.get("/samples/../../.env").status_code == 404


# ---------- middleware ----------

def test_process_time_header(client):
    assert "x-process-time-ms" in client.get("/health").headers


def test_docs_example_is_a_valid_wafer(client):
    example = AnalyzeRequest.model_fields["wafer_map"].examples[0]
    assert client.post("/analyze", json={"wafer_map": example}).status_code == 200

class FakeStore:
    def get(self, index):
        if index != 1:
            raise IndexError(index)
        return {"index": 1, "label": "Scratch", "wafer_map": [[1] * 20 for _ in range(20)]}


@pytest.fixture
def client(samples_dir):
    app = create_app(pipeline_factory=FakePipeline, samples_dir=samples_dir,
                     past_store_factory=FakeStore)
    with TestClient(app) as c:
        yield c

def test_past_wafer_found(client):
    body = client.get("/past-wafers/1").json()
    assert body["label"] == "Scratch" and len(body["wafer_map"]) == 20


def test_past_wafer_out_of_range_is_404(client):
    assert client.get("/past-wafers/999").status_code == 404


def test_past_wafers_unavailable_is_503(samples_dir):
    def broken():
        raise FileNotFoundError("no data")
    app = create_app(pipeline_factory=FakePipeline, samples_dir=samples_dir, past_store_factory=broken)
    with TestClient(app) as c:
        assert c.get("/past-wafers/1").status_code == 503

def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "WAFERLENS" in response.text


def test_static_files_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200