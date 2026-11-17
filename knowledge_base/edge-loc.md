---
pattern: Edge-Loc
title: Edge-Loc defect pattern
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

- A cluster of failed dies on one part of the wafer edge, not all the way around [3][4]
- Its position changes from wafer to wafer [2]
- Failed dies sit farthest from the wafer centre, which separates it from Loc [2]

## Common causes

- Often associated with irregular temperature during annealing or impurities from a load-lock valve [3]
- Commonly traced to handling contact or a local fault at the edge [4]

## What engineers usually check

- Handler and gripper logs [4]
- Thermal process settings, since Edge-Loc and Scratch call for different fixes: thermal tuning for Edge-Loc, mechanical handling checks for Scratch [3]

## Often confused with

- Loc: the two differ mainly in how close the cluster is to the edge, and expert labels between them are partly subjective [2]
- Scratch: both can appear as elongated clusters near the edge; mixing them up sends engineers to the wrong module [3]
- In WaferLens: 37 Loc wafers predicted as Edge-Loc, 32 Edge-Loc predicted as Loc, and 36 Edge-Loc predicted as none [5]