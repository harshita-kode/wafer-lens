"""Check the retriever returns the right notes for every pattern and for free-text questions."""
import sys

from src.rag.retriever import REQUIRED_SECTIONS, KnowledgeRetriever

# Plain-English questions that never use the class name, with the pattern that should win
QUESTIONS = [
    ("Why would failed dies form a thin line across the wafer?", "Scratch"),
    ("What causes failures all the way around the wafer rim?", "Edge-Ring"),
    ("Failures are concentrated in the middle of the wafer", "Center"),
    ("Almost every die on the wafer has failed", "Near-full"),
    ("A ring of failures with good dies in the very middle", "Donut"),
    ("Failed dies are scattered everywhere with no shape", "Random"),
    ("A cluster of failures on one small part of the edge", "Edge-Loc"),
    ("A wafer with only a few scattered failures and no pattern", "none"),
]


def check_patterns(r: KnowledgeRetriever) -> int:
    print("Per-pattern retrieval (filtered)")
    failures = 0
    for pattern in r.classes:
        chunks = r.retrieve(pattern, k=3, include_general=False)
        sections = [c["section"] for c in chunks]
        right_pattern = all(c["pattern"] == pattern for c in chunks)
        has_required = sections[:2] == REQUIRED_SECTIONS
        n_sources = sum(len(c["sources"]) for c in chunks)

        ok = right_pattern and has_required
        failures += not ok
        print(f"  [{'OK ' if ok else 'FIX'}] {pattern:10s} sections={sections}  citations={n_sources}")
    return failures


def check_questions(r: KnowledgeRetriever) -> None:
    print("\nFree-text search (no filter)")
    hit1 = hit3 = 0
    for question, expected in QUESTIONS:
        top = [c["pattern"] for c in r.search(question, n=3)]
        hit1 += top[0] == expected
        hit3 += expected in top
        mark = "OK " if top[0] == expected else ("~  " if expected in top else "MISS")
        print(f"  [{mark}] expected {expected:10s} got {top}")
    n = len(QUESTIONS)
    print(f"\n  Top-1 hit rate: {hit1}/{n} ({hit1 / n:.0%})")
    print(f"  Top-3 hit rate: {hit3}/{n} ({hit3 / n:.0%})")


def main() -> int:
    r = KnowledgeRetriever()
    failures = check_patterns(r)
    check_questions(r)
    print(f"\n{'All pattern checks passed' if failures == 0 else f'{failures} pattern check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())