import numpy as np

from src.agent.investigate import dihedral_variants, neighbour_shares, resolve

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
UNCERTAIN = {"label": "Loc", "confidence": 0.55, "uncertain": True,
             "top_k": [{"label": "Loc", "confidence": 0.55}, {"label": "Edge-Loc", "confidence": 0.30}]}


def probs(**kw):
    p = np.zeros(len(CLASSES))
    for name, value in kw.items():
        p[CLASSES.index(name.replace("_", "-"))] = value
    return p


def shares(*labels):
    return neighbour_shares([{"label": label} for label in labels], CLASSES)


def test_eight_distinct_variants():
    m = np.arange(12).reshape(3, 4)
    assert len(dihedral_variants(m)) == 8
    assert len({v.tobytes() + bytes(v.shape) for v in dihedral_variants(m)}) == 8


def test_neighbour_shares():
    s = shares("Edge-Loc", "Edge-Loc", "Edge-Loc", "Loc", "Scratch")
    assert s[CLASSES.index("Edge-Loc")] == 0.6


def test_agreeing_evidence_resolves():
    r = resolve(UNCERTAIN, probs(Loc=0.3, Edge_Loc=0.6),
                shares("Edge-Loc", "Edge-Loc", "Edge-Loc", "Edge-Loc", "Loc"), CLASSES)
    assert r["label"] == "Edge-Loc" and not r["uncertain"] and r["resolved_by"] == "investigation"


def test_conflicting_evidence_stays_uncertain():
    r = resolve(UNCERTAIN, probs(Loc=0.45, Edge_Loc=0.40),
                shares("Loc", "Loc", "Edge-Loc", "Edge-Loc", "Scratch"), CLASSES)
    assert r["uncertain"] and r["resolved_by"] == "unresolved"
    assert set(r["candidates"]) == {"Loc", "Edge-Loc"}


def test_without_neighbours_uses_tta_only():
    r = resolve(UNCERTAIN, probs(Loc=0.1, Edge_Loc=0.85), shares(), CLASSES)
    assert r["label"] == "Edge-Loc" and r["scores"]["Edge-Loc"] == 0.85