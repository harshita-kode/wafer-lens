"""Find similar past wafers using the CNN's 128-number fingerprints."""
import hashlib
import json
from collections import Counter

import chromadb
import numpy as np

from src.config import CHROMA_DIR, MODEL_PATH, SIMILAR_COLLECTION, SIMILAR_K

ADD_BATCH = 5000     # Chroma limits how many items one add() call can take


def normalize(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def model_fingerprint(model_path=MODEL_PATH) -> str:
    """Short hash of the model file. Changes whenever the model is retrained."""
    return hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]


def build_index(classifier, X: np.ndarray, y: np.ndarray, split: str = "train"):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(SIMILAR_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        SIMILAR_COLLECTION,
        metadata={"model_fingerprint": model_fingerprint(), "split": split},
    )

    print(f"Embedding {len(X)} {split} wafers...")
    embeddings = normalize(classifier.embed_batch(list(X)))

    for start in range(0, len(X), ADD_BATCH):
        end = min(start + ADD_BATCH, len(X))
        collection.add(
            ids=[f"{split}-{i}" for i in range(start, end)],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[{"label": classifier.classes[int(y[i])], "split": split, "index": i}
                       for i in range(start, end)],
        )
    return collection


def _to_neighbours(result: dict, row: int = 0) -> list[dict]:
    return [
        {"id": i, "label": m["label"], "index": m["index"], "similarity": round(1 - d / 2, 4)}
        for i, m, d in zip(result["ids"][row], result["metadatas"][row], result["distances"][row])
    ]


class SimilarWaferIndex:
    def __init__(self, check_model: bool = True):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = client.get_collection(SIMILAR_COLLECTION)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{SIMILAR_COLLECTION}' not found. Run: python -m src.similar.index"
            ) from e

        if check_model:
            stored = (self.collection.metadata or {}).get("model_fingerprint")
            if stored != model_fingerprint():
                raise ValueError("Similar-wafer index was built with a different model. "
                                 "Rebuild it: python -m src.similar.index")

    def find(self, embedding, k: int = SIMILAR_K) -> list[dict]:
        result = self.collection.query(
            query_embeddings=normalize(embedding).reshape(1, -1).tolist(), n_results=k)
        return _to_neighbours(result)


_default: SimilarWaferIndex | None = None


def get_similar_index() -> SimilarWaferIndex:
    global _default
    if _default is None:
        _default = SimilarWaferIndex()
    return _default


def evaluate(index: SimilarWaferIndex, classifier, X, y, n: int = 1000,
             k: int = SIMILAR_K, seed: int = 42) -> dict:
    """precision@k and majority-vote accuracy of the neighbours, on held-out wafers."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(X), size=min(n, len(X)), replace=False)
    embeddings = normalize(classifier.embed_batch(list(X[picks])))

    agreement, vote_correct = [], 0
    for start in range(0, len(picks), 500):
        result = index.collection.query(
            query_embeddings=embeddings[start:start + 500].tolist(), n_results=k)
        for row in range(len(result["ids"])):
            true = classifier.classes[int(y[picks[start + row]])]
            labels = [m["label"] for m in result["metadatas"][row]]
            agreement.append(sum(label == true for label in labels) / k)
            vote_correct += Counter(labels).most_common(1)[0][0] == true

    return {"k": k, "wafers": len(picks),
            "precision_at_k": round(float(np.mean(agreement)), 4),
            "knn_vote_accuracy": round(vote_correct / len(picks), 4)}


if __name__ == "__main__":
    from src.classifier.predict import get_classifier
    from src.config import DATA_DIR, PROJECT_ROOT

    data = np.load(DATA_DIR / "processed" / "wafers_64.npz")
    clf = get_classifier()

    build_index(clf, data["X_train"], data["y_train"], split="train")
    index = SimilarWaferIndex()
    print(f"Indexed {index.collection.count()} past wafers")

    example = data["X_test"][0]
    print(f"\nNeighbours of test wafer 0 (true: {clf.classes[data['y_test'][0]]}):")
    for nb in index.find(clf.embed(example)):
        print(f"  {nb['similarity']:.3f}  {nb['label']:10s}  train #{nb['index']}")

    metrics = evaluate(index, clf, data["X_test"], data["y_test"])
    print(f"\nprecision@{metrics['k']}:     {metrics['precision_at_k']:.1%}")
    print(f"kNN vote accuracy: {metrics['knn_vote_accuracy']:.1%}  (CNN test accuracy: 89.4%)")

    out = PROJECT_ROOT / "results" / "similar_eval.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {out.relative_to(PROJECT_ROOT)}")