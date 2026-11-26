import numpy as np
import pytest

from src.config import DATA_DIR, MODEL_PATH, UNCERTAIN_THRESHOLD

TEST_DATA = DATA_DIR / "processed" / "wafers_64.npz"

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(),
                                reason="Trained model missing (models/wafer_cnn.pt)")


@pytest.fixture(scope="module")
def clf():
    from src.classifier.predict import WaferClassifier
    return WaferClassifier(device="cpu")


def round_wafer(rows=40, cols=40, center_fail=True):
    """A simple round wafer: 0 outside, 1 inside, optional failed blob in the middle."""
    y, x = np.mgrid[0:rows, 0:cols]
    r = np.hypot((y - (rows - 1) / 2) / (rows / 2), (x - (cols - 1) / 2) / (cols / 2))
    wafer = np.where(r <= 1, 1, 0).astype(np.uint8)
    if center_fail:
        wafer[r < 0.3] = 2
    return wafer


def test_prediction_has_expected_structure(clf):
    result = clf.predict(round_wafer())
    assert result["label"] in clf.classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_k"]) == 3
    assert result["top_k"][0]["label"] == result["label"]


def test_top_k_is_sorted_and_sums_below_one(clf):
    confs = [t["confidence"] for t in clf.predict(round_wafer())["top_k"]]
    assert confs == sorted(confs, reverse=True)
    assert sum(confs) <= 1.0 + 1e-6


def test_uncertain_flag_follows_threshold(clf):
    result = clf.predict(round_wafer())
    assert result["uncertain"] == (result["confidence"] < UNCERTAIN_THRESHOLD)


def test_predictions_are_deterministic(clf):
    wafer = round_wafer()
    assert clf.predict(wafer) == clf.predict(wafer)


@pytest.mark.parametrize("shape", [(25, 25), (64, 64), (100, 80)])
def test_any_wafer_size_is_accepted(clf, shape):
    assert clf.predict(round_wafer(*shape))["label"] in clf.classes


def test_batch_matches_single(clf):
    wafers = [round_wafer(), round_wafer(center_fail=False), round_wafer(50, 50)]
    batch = clf.predict_batch(wafers)
    singles = [clf.predict(w) for w in wafers]
    assert [b["label"] for b in batch] == [s["label"] for s in singles]


def test_embedding_shape_and_values(clf):
    emb = clf.embed(round_wafer())
    assert emb.shape == (128,)
    assert (emb >= 0).all()            # the embedding layer ends in ReLU


@pytest.mark.parametrize("bad", [np.zeros((5, 5)), np.full((30, 30), 7), np.zeros(30)])
def test_invalid_wafers_raise_value_error(clf, bad):
    with pytest.raises(ValueError):
        clf.predict(bad)


def test_preprocess_matches_training_format():
    from src.classifier.preprocess import preprocess
    rng = np.random.default_rng(0)
    tensor = preprocess(rng.integers(0, 3, size=(33, 41)))
    assert tuple(tensor.shape) == (1, 64, 64)
    assert set(np.unique(tensor.numpy()).tolist()) <= {0.0, 0.5, 1.0}


@pytest.mark.skipif(not TEST_DATA.exists(), reason="Processed test data missing")
def test_accuracy_on_held_out_sample(clf):
    data = np.load(TEST_DATA)
    X, y = data["X_test"], data["y_test"]
    picks = np.random.default_rng(42).choice(len(X), size=1000, replace=False)
    preds = clf.predict_batch(list(X[picks]))
    accuracy = np.mean([clf.classes.index(p["label"]) == y[i] for p, i in zip(preds, picks)])
    assert accuracy >= 0.85, f"accuracy dropped to {accuracy:.1%}"