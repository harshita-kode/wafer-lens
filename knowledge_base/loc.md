---
pattern: Loc
title: Loc (local) defect pattern
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

- A cluster of failed dies inside the wafer, away from the edge [3]
- Failed dies tend to sit at a medium distance from the centre [2]
- Its position changes from wafer to wafer [2]

## Common causes

- Often associated with vacuum-pressure differences from slit-valve leaks, poor pump operation, or vibration [3]

## What engineers usually check

- Start from the process steps named under common causes, then confirm against the tool history of affected wafers [4]
- Check whether the pattern repeats across wafers or lots before acting [4]

## Often confused with

- Edge-Loc: the main difference is whether the cluster touches the edge [2]
- Scratch: a short or broken scratch can look like a small cluster [2]
- Loc is hard for many models, not just WaferLens [3]
- In WaferLens: 40 Scratch predicted as Loc, 37 Loc predicted as Edge-Loc, 32 Loc predicted as Scratch, and 31 Loc predicted as none; recall is 0.71 [5]