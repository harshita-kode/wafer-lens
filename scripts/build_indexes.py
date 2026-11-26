"""Build the knowledge-base and past-wafer indexes in one step (used by the Docker build)."""
import time

import numpy as np


def main() -> None:
    from src.classifier.predict import get_classifier
    from src.config import PAST_WAFERS_PATH
    from src.rag.ingest import build_index as build_kb_index
    from src.rag.ingest import load_all_chunks
    from src.similar.index import SimilarWaferIndex
    from src.similar.index import build_index as build_similar_index

    start = time.time()

    kb = build_kb_index(load_all_chunks())
    print(f"Knowledge base: {kb.count()} chunks")

    data = np.load(PAST_WAFERS_PATH)
    build_similar_index(get_classifier(), data["X_train"], data["y_train"], split="train")
    print(f"Past wafers: {SimilarWaferIndex().collection.count()} fingerprints")

    print(f"Indexes built in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()