import json

import numpy as np
import torch

from src.config import LABEL_MAP_PATH, MODEL_PATH, UNCERTAIN_THRESHOLD
from src.classifier.model import WaferCNN
from src.classifier.preprocess import preprocess


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WaferClassifier:
    """Loads the trained CNN once and predicts wafer defect patterns."""

    def __init__(self, model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH,
                 threshold: float = UNCERTAIN_THRESHOLD, device: str | None = None):
        self.device = device or _pick_device()
        self.threshold = threshold

        with open(label_map_path) as f:
            self.classes = json.load(f)

        ckpt = torch.load(model_path, map_location=self.device)
        if ckpt["classes"] != self.classes:
            raise ValueError("Class order in checkpoint does not match label_map.json")

        self.img_size = ckpt.get("img_size", 64)
        self.model_epoch = ckpt.get("epoch")

        self.model = WaferCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ---------- internal helpers ----------

    def _to_batch(self, wafer_maps) -> torch.Tensor:
        return torch.stack([preprocess(m, self.img_size) for m in wafer_maps]).to(self.device)

    def _format(self, probs: np.ndarray, top_k: int) -> dict:
        order = np.argsort(probs)[::-1]           # class indices, highest probability first
        best = int(order[0])
        confidence = float(probs[best])
        return {
            "label": self.classes[best],
            "confidence": round(confidence, 4),
            "uncertain": confidence < self.threshold,
            "top_k": [
                {"label": self.classes[i], "confidence": round(float(probs[i]), 4)}
                for i in order[:top_k]
            ],
        }

    # ---------- public API ----------

    @torch.no_grad()
    def predict_batch(self, wafer_maps, top_k: int = 3, batch_size: int = 256) -> list[dict]:
        results = []
        for start in range(0, len(wafer_maps), batch_size):
            chunk = wafer_maps[start:start + batch_size]
            probs = torch.softmax(self.model(self._to_batch(chunk)), dim=1).cpu().numpy()
            results.extend(self._format(p, top_k) for p in probs)
        return results

    def predict(self, wafer_map, top_k: int = 3) -> dict:
        return self.predict_batch([wafer_map], top_k)[0]

    @torch.no_grad()
    def embed_batch(self, wafer_maps, batch_size: int = 256) -> np.ndarray:
        """128-number fingerprints, used later to find similar past wafers."""
        parts = []
        for start in range(0, len(wafer_maps), batch_size):
            chunk = wafer_maps[start:start + batch_size]
            parts.append(self.model.embed(self._to_batch(chunk)).cpu().numpy())
        return np.concatenate(parts)

    def embed(self, wafer_map) -> np.ndarray:
        return self.embed_batch([wafer_map])[0]
    
    @torch.no_grad()
    def predict_proba_batch(self, wafer_maps, batch_size: int = 256) -> np.ndarray:
        """All class probabilities, shape (n_wafers, n_classes)."""
        parts = []
        for start in range(0, len(wafer_maps), batch_size):
            chunk = wafer_maps[start:start + batch_size]
            parts.append(torch.softmax(self.model(self._to_batch(chunk)), dim=1).cpu().numpy())
        return np.concatenate(parts)

    def predict_batch(self, wafer_maps, top_k: int = 3, batch_size: int = 256) -> list[dict]:
        return [self._format(p, top_k) for p in self.predict_proba_batch(wafer_maps, batch_size)]


# ---------- simple module-level shortcut ----------

_default: WaferClassifier | None = None


def get_classifier() -> WaferClassifier:
    """Create the classifier on first use, then reuse it."""
    global _default
    if _default is None:
        _default = WaferClassifier()
    return _default


def predict(wafer_map, top_k: int = 3) -> dict:
    return get_classifier().predict(wafer_map, top_k)

if __name__ == "__main__":
    import time
    from src.config import DATA_DIR
    from src.classifier.preprocess import resize_wafer

    start = time.time()
    clf = WaferClassifier()
    print(f"Loaded model (epoch {clf.model_epoch}) on {clf.device} in {time.time() - start:.1f}s\n")

    data = np.load(DATA_DIR / "processed" / "wafers_64.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    # 1. A few single predictions
    print("Single predictions:")
    for i in range(5):
        r = clf.predict(X_test[i])
        flag = "  (uncertain)" if r["uncertain"] else ""
        print(f"  true={clf.classes[y_test[i]]:10s} pred={r['label']:10s} {r['confidence']:.0%}{flag}")

    # 2. Whole test set: must match the notebook (89.4%)
    start = time.time()
    results = clf.predict_batch(list(X_test))
    preds = np.array([clf.classes.index(r["label"]) for r in results])
    print(f"\nTest accuracy via predict.py: {(preds == y_test).mean():.1%} "
          f"({len(X_test)} wafers in {time.time() - start:.1f}s)")

    # 3. A wafer of a different size still works
    odd = resize_wafer(X_test[0], 41)
    r = clf.predict(odd)
    print(f"\n41x41 wafer -> {r['label']} ({r['confidence']:.0%})")

    # 4. Bad input is rejected clearly
    for bad in [np.zeros((5, 5)), np.full((30, 30), 7), np.zeros(30)]:
        try:
            clf.predict(bad)
        except ValueError as e:
            print(f"Rejected as expected: {e}")

    # 5. Fingerprint shape
    print(f"\nEmbedding shape: {clf.embed(X_test[0]).shape}")