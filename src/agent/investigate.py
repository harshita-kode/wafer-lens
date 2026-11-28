"""Gather extra evidence for uncertain predictions and decide between the top two candidates."""
import numpy as np

from src.config import AGENT_MIN_MARGIN, AGENT_MIN_SCORE


def dihedral_variants(wafer_map) -> list[np.ndarray]:
    """The 8 rotations/mirror images of a wafer (the same transforms used in training)."""
    m = np.asarray(wafer_map)
    rotations = [np.rot90(m, k) for k in range(4)]
    return [np.ascontiguousarray(v) for v in rotations + [np.fliplr(v) for v in rotations]]


def tta_probabilities(classifier, wafer_map) -> np.ndarray:
    """Average class probabilities over the 8 variants."""
    return classifier.predict_proba_batch(dihedral_variants(wafer_map)).mean(axis=0)


def neighbour_shares(similar: list[dict], classes: list[str]) -> np.ndarray:
    """Fraction of similar past wafers carrying each label."""
    counts = np.zeros(len(classes))
    for wafer in similar:
        counts[classes.index(wafer["label"])] += 1
    return counts / len(similar) if similar else counts


def resolve(prediction: dict, tta: np.ndarray, shares: np.ndarray, classes: list[str],
            min_score: float = AGENT_MIN_SCORE, min_margin: float = AGENT_MIN_MARGIN) -> dict:
    """Combine the evidence for the CNN's top two candidates and decide if one clearly wins."""
    candidates = [t["label"] for t in prediction["top_k"][:2]]
    w_neighbours = 0.5 if shares.sum() > 0 else 0.0      # no neighbours: rely on TTA alone

    scores = {}
    for c in candidates:
        i = classes.index(c)
        scores[c] = round(float((1 - w_neighbours) * tta[i] + w_neighbours * shares[i]), 4)

    ranked = sorted(candidates, key=scores.get, reverse=True)
    margin = round(scores[ranked[0]] - scores[ranked[1]], 4)
    resolved = scores[ranked[0]] >= min_score and margin >= min_margin

    return {"label": ranked[0], "uncertain": not resolved,
            "resolved_by": "investigation" if resolved else "unresolved",
            "candidates": ranked, "scores": scores, "margin": margin,
            "tta": {c: round(float(tta[classes.index(c)]), 4) for c in candidates},
            "neighbours": {c: round(float(shares[classes.index(c)]), 4) for c in candidates}}