"""Measure the investigation step on uncertain wafers: tune thresholds on val, report on test.

  python -m scripts.eval_agent                 # tuning grid on validation
  python -m scripts.eval_agent --split test    # final numbers with the configured thresholds
"""
import argparse
import json

import numpy as np

from src.agent.investigate import neighbour_shares, resolve, tta_probabilities
from src.classifier.predict import get_classifier
from src.config import (AGENT_MIN_MARGIN, AGENT_MIN_SCORE, DATA_DIR, PROJECT_ROOT, SIMILAR_K)
from src.similar.index import SimilarWaferIndex, normalize


def gather(split: str):
    """For every uncertain wafer in the split: (CNN prediction, TTA probs, neighbour shares, true label)."""
    data = np.load(DATA_DIR / "processed" / "wafers_64.npz")
    X, y = data[f"X_{split}"], data[f"y_{split}"]
    clf, index = get_classifier(), SimilarWaferIndex()

    preds = clf.predict_batch(list(X))
    picks = [i for i, p in enumerate(preds) if p["uncertain"]]
    embeddings = normalize(clf.embed_batch(list(X[picks])))

    rows = []
    for start in range(0, len(picks), 500):
        result = index.collection.query(query_embeddings=embeddings[start:start + 500].tolist(),
                                        n_results=SIMILAR_K)
        for r, metas in enumerate(result["metadatas"]):
            i = picks[start + r]
            rows.append((preds[i], tta_probabilities(clf, X[i]),
                         neighbour_shares(metas, clf.classes), clf.classes[int(y[i])]))
    return rows, len(X), clf.classes


def score(rows, classes, min_score, min_margin) -> dict:
    n = len(rows)
    cnn_correct = agent_correct = resolved = resolved_correct = cnn_correct_on_resolved = 0
    unresolved = unresolved_covered = 0

    for pred, tta, shares, true in rows:
        d = resolve(pred, tta, shares, classes, min_score, min_margin)
        cnn_correct += pred["label"] == true
        agent_correct += d["label"] == true
        if d["uncertain"]:
            unresolved += 1
            unresolved_covered += true in d["candidates"]
        else:
            resolved += 1
            resolved_correct += d["label"] == true
            cnn_correct_on_resolved += pred["label"] == true

    pct = lambda a, b: round(a / b, 4) if b else None
    return {"min_score": min_score, "min_margin": min_margin, "uncertain_wafers": n,
            "cnn_top1_accuracy": pct(cnn_correct, n),
            "agent_top1_accuracy": pct(agent_correct, n),
            "resolved_rate": pct(resolved, n),
            "resolved_accuracy": pct(resolved_correct, resolved),
            "cnn_accuracy_on_resolved": pct(cnn_correct_on_resolved, resolved),
            "unresolved_top2_coverage": pct(unresolved_covered, unresolved)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    rows, total, classes = gather(args.split)
    print(f"{args.split}: {len(rows)} of {total} wafers are uncertain\n")

    if args.split == "val":
        print(f"{'score':>6} {'margin':>7} {'resolved':>9} {'res.acc':>8} {'cnn@res':>8} {'agent':>7} {'cnn':>6}")
        for min_score in (0.5, 0.6, 0.7):
            for min_margin in (0.1, 0.2, 0.3):
                m = score(rows, classes, min_score, min_margin)
                print(f"{min_score:>6} {min_margin:>7} {m['resolved_rate']:>9.0%} "
                      f"{(m['resolved_accuracy'] or 0):>8.0%} {(m['cnn_accuracy_on_resolved'] or 0):>8.0%} "
                      f"{m['agent_top1_accuracy']:>7.0%} {m['cnn_top1_accuracy']:>6.0%}")
        return

    m = score(rows, classes, AGENT_MIN_SCORE, AGENT_MIN_MARGIN)
    for key, value in m.items():
        print(f"  {key:28s} {value}")
    out = PROJECT_ROOT / "results" / "agent_eval.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()