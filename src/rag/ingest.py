"""Load knowledge-base notes, split them into chunks, embed them, store them in ChromaDB."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import frontmatter
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, EMBED_MODEL, KB_COLLECTION, KB_DIR


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def slugify(text: str) -> str:
    """'What engineers usually check' -> 'what-engineers-usually-check'"""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split Markdown at each '## ' heading -> [(heading, text), ...]"""
    parts = re.split(r"^## ", body, flags=re.MULTILINE)
    sections = []
    for part in parts[1:]:                      # parts[0] is anything before the first heading
        heading, _, text = part.partition("\n")
        text = text.strip()
        if text:
            sections.append((heading.strip(), text))
    return sections


def note_to_chunks(path: Path) -> list[Chunk]:
    note = frontmatter.load(path)
    meta = note.metadata
    sources = {s["id"]: s for s in meta.get("sources", [])}

    chunks = []
    for heading, text in split_sections(note.content):
        cited_ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        cited = [sources[i] for i in cited_ids if i in sources]

        chunks.append(Chunk(
            id=f"{path.stem}::{slugify(heading)}",
            text=f"{meta['title']} — {heading}\n\n{text}",
            metadata={
                "pattern": meta["pattern"],
                "section": heading,
                "title": meta["title"],
                "file": path.name,
                "sources": json.dumps(cited),   # Chroma metadata must be plain text/numbers
            },
        ))
    return chunks


def load_all_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(note_to_chunks(path))
    return chunks


def build_index(chunks: list[Chunk]) -> chromadb.Collection:
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    embeddings = model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(KB_COLLECTION)   # start fresh every time
    except Exception:
        pass
    collection = client.create_collection(
        KB_COLLECTION, metadata={"embedding_model": EMBED_MODEL}
    )

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
    )
    return collection


def smoke_test(collection: chromadb.Collection) -> None:
    """Search WITHOUT a filter, to check the meaning-search works on its own."""
    model = SentenceTransformer(EMBED_MODEL)
    question = "Why do linear trails of failed dies appear?"
    q_emb = model.encode([question], normalize_embeddings=True).tolist()

    res = collection.query(query_embeddings=q_emb, n_results=3)
    print(f'\nTest question: "{question}"')
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        similarity = 1 - dist / 2
        print(f"  {similarity:.2f}  {meta['pattern']:10s} | {meta['section']}")


def main() -> None:
    chunks = load_all_chunks()
    n_notes = len({c.metadata["file"] for c in chunks})
    print(f"Found {len(chunks)} chunks from {n_notes} notes")

    collection = build_index(chunks)
    print(f"Stored {collection.count()} chunks in {CHROMA_DIR.name}/{KB_COLLECTION}")

    smoke_test(collection)


if __name__ == "__main__":
    main()