"""Simple, deterministic measurements of one wafer map."""
import numpy as np

from src.classifier.preprocess import validate_wafer

ZONES = {"center": (0.0, 0.35), "middle": (0.35, 0.75), "edge": (0.75, 10.0)}


def wafer_facts(wafer_map) -> dict:
    m = validate_wafer(wafer_map)
    dies, fails = m > 0, m == 2
    n_dies, n_fails = int(dies.sum()), int(fails.sum())

    rows, cols = m.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    r = np.hypot((yy - (rows - 1) / 2) / (rows / 2), (xx - (cols - 1) / 2) / (cols / 2))

    share = {zone: round(float((fails & (r >= lo) & (r < hi)).sum() / n_fails), 3) if n_fails else 0.0
             for zone, (lo, hi) in ZONES.items()}

    return {"dies": n_dies, "failed_dies": n_fails,
            "fail_rate": round(n_fails / n_dies, 4) if n_dies else 0.0,
            "failure_share_by_zone": share}