import numpy as np

from src.agent.facts import wafer_facts


def ring_wafer(size=40):
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot((y - (size - 1) / 2) / (size / 2), (x - (size - 1) / 2) / (size / 2))
    w = np.where(r <= 1, 1, 0).astype(np.uint8)
    w[(r > 0.8) & (r <= 1)] = 2
    return w


def test_counts_and_fail_rate():
    f = wafer_facts(ring_wafer())
    assert f["dies"] > f["failed_dies"] > 0
    assert abs(f["fail_rate"] - f["failed_dies"] / f["dies"]) < 1e-3


def test_edge_ring_failures_sit_in_edge_zone():
    assert wafer_facts(ring_wafer())["failure_share_by_zone"]["edge"] > 0.9


def test_clean_wafer_has_no_failures():
    w = ring_wafer()
    w[w == 2] = 1
    f = wafer_facts(w)
    assert f["failed_dies"] == 0 and f["fail_rate"] == 0.0