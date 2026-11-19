"""Find the knowledge-base chunks that explain a predicted defect pattern."""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, EMBED_MODEL, KB_COLLECTION, LABEL_MAP_PATH

REQUIRED_SECTIONS = ["Common causes", "What engineers usually check"]
DEFAULT_QUERY = "What causes the {pattern} wafer defect pattern and what should engineers check?"


def _to_chunk(chunk_id: str, text: str, meta: dict, distance: float) -> dict:
    return {
        "id": chunk_id,
        "pattern": meta["pattern"],
        "section": meta["section"],
        "title": meta["title"],
        "text": text,
        "sources": json.loads(meta.get("sources", "[]")),
        "similarity": round(1 - distance / 2, 4),
    }


def _parse(result: dict) -> list[dict]:
    """Chroma returns lists-of-lists (one list per query). We always send one query."""
    return [
        _to_chunk(i, d, m, dist)
        for i, d, m, dist in zip(result["ids"][0], result["documents"][0],
                                 result["metadatas"][0], result["distances"][0])
    ]


def collect_sources(chunks: list[dict]) -> list[dict]:
    """Unique sources across all chunks, in the order they first appear."""
    seen, sources = set(), []
    for chunk in chunks:
        for s in chunk["sources"]:
            key = s.get("url") or s.get("title")
            if key not in seen:
                seen.add(key)
                sources.append(s)
    return sources


class KnowledgeRetriever:
    """Loads the embedding model and the Chroma collection once, then answers many searches."""

    def __init__(self, chroma_dir=CHROMA_DIR, collection_name=KB_COLLECTION,
                 model_name=EMBED_MODEL):
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in {chroma_dir}. "
                "Run: python -m src.rag.ingest"
            ) from e

        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != model_name:
            raise ValueError(
                f"Index was built with '{stored_model}' but the retriever uses "
                f"'{model_name}'. Re-run: python -m src.rag.ingest"
            )

        self.model = SentenceTransformer(model_name)
        with open(LABEL_MAP_PATH) as f:
            self.classes = json.load(f)

    # ---------- internal helpers ----------

    def _embed(self, text: str) -> list[list[float]]:
        return self.model.encode([text], normalize_embeddings=True).tolist()

    def _count(self, pattern: str) -> int:
        return len(self.collection.get(where={"pattern": pattern}, include=["metadatas"])["ids"])

    def _search_pattern(self, pattern: str, query: str) -> list[dict]:
        """All chunks of one pattern, ranked by similarity to the query."""
        n = self._count(pattern)
        if n == 0:
            return []
        result = self.collection.query(
            query_embeddings=self._embed(query),
            n_results=n,
            where={"pattern": pattern},
        )
        return _parse(result)

    # ---------- public API ----------

    def search(self, query: str, n: int = 3) -> list[dict]:
        """Free-text search across ALL notes, no filter."""
        return _parse(self.collection.query(query_embeddings=self._embed(query), n_results=n))

    def retrieve(self, pattern: str, k: int = 3, query: str | None = None,
                 include_general: bool = True) -> list[dict]:
        """Best k chunks for one pattern (required sections first), plus one general chunk."""
        if pattern not in self.classes:
            raise ValueError(f"Unknown pattern '{pattern}'. Expected one of {self.classes}")

        query = query or DEFAULT_QUERY.format(pattern=pattern)
        ranked = self._search_pattern(pattern, query)

        required = [c for section in REQUIRED_SECTIONS for c in ranked if c["section"] == section]
        others = [c for c in ranked if c["section"] not in REQUIRED_SECTIONS]
        chunks = (required + others)[:k]

        if include_general:
            chunks += self._search_pattern("general", query)[:1]
        return chunks


if __name__ == "__main__":
    r = KnowledgeRetriever()

    chunks = r.retrieve("Edge-Ring")
    print("retrieve('Edge-Ring'):")
    for c in chunks:
        print(f"  {c['similarity']:.2f}  {c['pattern']:10s} | {c['section']}")

    print("\nSources to cite:")
    for s in collect_sources(chunks):
        print(f"  - {s['authors']} ({s['year']}). {s['title']}")

    print("\nFree-text search: 'failures along a thin line across the wafer'")
    for c in r.search("failures along a thin line across the wafer"):
        print(f"  {c['similarity']:.2f}  {c['pattern']:10s} | {c['section']}")