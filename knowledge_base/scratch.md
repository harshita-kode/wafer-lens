---
pattern: Scratch
title: Scratch defect pattern
sources:
  - id: 2
    title: "Development of a Wafer Defect Pattern Classifier Using Polar Coordinate System Transformed Inputs and Convolutional Neural Networks"
    authors: "Moo Hyun Kim, Tae Seon Kim"
    year: 2024
    url: "https://doi.org/10.3390/electronics13071360"
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

- A connected line or trail of failed dies, straight or curved [3][4]
- The thinnest of the defect patterns, so it can lose its shape after heavy noise filtering [2]
- Its position changes from wafer to wafer [2]

## Common causes

- Often linked to mechanical damage from transfer robots, wafer handling, or chemical-mechanical polishing (CMP) [3]
- Commonly traced to handling and CMP [4]

## What engineers usually check

- Handler movement paths and the CMP tool [4]
- Mechanical handling, rather than thermal settings, which is the key difference from Edge-Loc [3]

## Often confused with

- Loc: other CNN studies also report scratches being classified as Loc [2]
- Edge-Loc: both can look like elongated clusters near the edge [3]
- Scratch is the weakest WaferLens class (recall 0.55): 40 predicted as Loc and 29 as none; thin lines can break up when large wafers are resized to 64x64 [5]