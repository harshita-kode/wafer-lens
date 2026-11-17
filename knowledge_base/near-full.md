---
pattern: Near-full
title: Near-full defect pattern
sources:
  - id: 3
    title: "Bridging Quantum Computing Paradigms toward Semiconductor Yield: A Controlled CV-versus-DV Comparison on Wafer-Map Defect Classification"
    authors: "Yeonhong Kim, Jonghyeok Im, Monu Nath Baitha, Kyoungsik Kim"
    year: 2026
    url: "https://arxiv.org/abs/2607.00961"
  - id: 4
    title: "Wafer Defect Map Analysis: How to Read Spatial Patterns and Find Root Causes"
    authors: "Averroes"
    year: 2026
    url: "https://averroes.ai/blog/wafer-defect-map-analysis"
  - id: 5
    title: "WaferLens CNN test-set evaluation (results/test_metrics.json)"
    authors: "Own project"
    year: 2026
    url: "results/test_metrics.json"
---

## What it looks like

- Failed dies covering nearly the entire wafer [3]

## Common causes

- Often associated with photoresist rupture caused by electron overcharging during plasma ion implantation [3]

## What engineers usually check

- Start from the process step named under common causes, then confirm against the tool history of affected wafers [4]
- Check whether other wafers in the same lot show the same pattern [4]

## Often confused with

- Random: both are spread across the wafer, but near-full affects almost every die
- WaferLens caught every real near-full wafer in testing (recall 1.00), but a third of its near-full predictions were wrong (precision 0.67); only 22 test examples exist, so these numbers are less reliable [5]