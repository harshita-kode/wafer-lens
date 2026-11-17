---
pattern: Random
title: Random defect pattern
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

- Failed dies scattered across the wafer with no clear cluster or shape [3]
- Much denser than the light background noise found on most normal wafers [2]

## Common causes

- Often associated with abnormal vacuum, gas, or fluid eruption during processing [3]

## What engineers usually check

- Start from the process steps named under common causes, then confirm against the tool history of affected wafers [4]
- Check whether the pattern repeats across wafers or lots, to separate a real excursion from a one-off [4]

## Often confused with

- none: normal wafers also carry some scattered failures, from particles and the process environment [2][4]
- Near-full: both spread across the wafer, but near-full covers almost every die
- WaferLens finds almost all random wafers (recall 0.97) but over-predicts the class (precision 0.74) [5]