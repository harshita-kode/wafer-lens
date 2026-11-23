"""Run test wafers through classifier -> retriever -> writer and score the reports."""
import json
import logging
import re
import time

import numpy as np

from src.classifier.predict import get_classifier
from src.config import DATA_DIR, PROJECT_ROOT
from src.llm.client import get_llm
from src.rag.retriever import get_retriever
from src.report.writer import write_report

PER_CLASS = 2
OVERCONFIDENT = ["is caused by", "are caused by", "definitely", "always caused", "certainly"]

def expected_sections(report, prediction) -> list[str]:
    if len(report.patterns) > 1:
        return ["What this usually means", "Suggested checks", "How to tell them apart"]
    if prediction["label"] == "none":
        return ["What this usually means"]
    return ["What this usually means", "Suggested checks"]


def body_of(markdown: str) -> str:
    """The LLM-written part: everything after the header line, before Sources."""
    without_sources = markdown.split("**Sources:**")[0]
    return without_sources.split("\n", 1)[1] if "\n" in without_sources else ""


def score(report, prediction) -> dict:
    body = body_of(report.markdown)
    bullets = [line for line in body.splitlines() if line.strip().startswith(("- ", "* "))]
    cited_bullets = [b for b in bullets if re.search(r"\[S\d+\]", b)]
    found_phrases = [p for p in OVERCONFIDENT if p in body.lower()]
    both_mentioned = all(p in body for p in report.patterns) if len(report.patterns) > 1 else None

    return {
        "llm_used": report.generated_by == "llm",
        "citations": len(re.findall(r"\[S\d+\]", body)),
        "bullets": len(bullets),
        "bullet_citation_rate": round(len(cited_bullets) / len(bullets), 2) if bullets else 0.0,
        "overconfident_phrases": found_phrases,
        "uncertain_mentions_both": both_mentioned,
        "sections_ok": all(f"**{s}:**" in body for s in expected_sections(report, prediction)),
        "warnings": report.warnings,
    }


def pick_samples(y: np.ndarray, n_classes: int, per_class: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    picks = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        picks += rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist()
    return picks


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    data = np.load(DATA_DIR / "processed" / "wafers_64.npz")
    X, y = data["X_test"], data["y_test"]

    clf, retriever, llm = get_classifier(), get_retriever(), get_llm()
    rows = []

    for i in pick_samples(y, len(clf.classes), PER_CLASS):
        hits_before = getattr(llm, "hits", 0)
        start = time.time()
        pred = clf.predict(X[i])
        report = write_report(pred, retriever.retrieve_for_prediction(pred), llm=llm)
        cached = getattr(llm, "hits", 0) > hits_before
        row = {
            "index": int(i),
            "true": clf.classes[y[i]],
            "pred": pred["label"],
            "confidence": pred["confidence"],
            "uncertain": pred["uncertain"],
            "seconds": round(time.time() - start, 2),
            "cached": cached,
            **score(report, pred),
        }
        rows.append(row)
        flag = "OK " if row["llm_used"] and row["sections_ok"] \
               and row["bullet_citation_rate"] >= 0.8 and not row["overconfident_phrases"] else "CHK"
        print(f"[{flag}] true={row['true']:10s} pred={row['pred']:10s} "
              f"cites={row['citations']:2d} cited-bullets={row['bullet_citation_rate']:.0%} "
              f"{row['seconds']:.1f}s")

    n = len(rows)
    uncertain_rows = [r for r in rows if r["uncertain_mentions_both"] is not None]
    summary = {
        "reports": n,
        "llm_used_rate": round(sum(r["llm_used"] for r in rows) / n, 2),
        "avg_citations": round(sum(r["citations"] for r in rows) / n, 1),
        "avg_bullet_citation_rate": round(sum(r["bullet_citation_rate"] for r in rows) / n, 2),
        "reports_with_overconfident_wording": sum(bool(r["overconfident_phrases"]) for r in rows),
        "uncertain_reports": len(uncertain_rows),
        "uncertain_mention_both_rate": round(
            sum(r["uncertain_mentions_both"] for r in uncertain_rows) / len(uncertain_rows), 2
        ) if uncertain_rows else None,
        "reports_with_warnings": sum(bool(r["warnings"]) for r in rows),
                "sections_ok_rate": round(sum(r["sections_ok"] for r in rows) / n, 2),
        "avg_seconds_uncached": round(
            np.mean([r["seconds"] for r in rows if not r["cached"]]), 2
        ) if any(not r["cached"] for r in rows) else None,
        "avg_seconds_cached": round(
            np.mean([r["seconds"] for r in rows if r["cached"]]), 2
        ) if any(r["cached"] for r in rows) else None,
    }
    if hasattr(llm, "hits"):
        summary["cache_hits"], summary["cache_misses"] = llm.hits, llm.misses

    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key:36s} {value}")

    out = PROJECT_ROOT / "results" / "report_eval.json"
    out.write_text(json.dumps({"summary": summary, "reports": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()