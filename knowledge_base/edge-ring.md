---
pattern: Edge-Ring
title: Edge-Ring defect pattern
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

- Failed dies along the whole outer edge of the wafer [2][3]
- A band that goes around the full circumference [4]

## Common causes

- Often associated with abnormal temperature control during rapid thermal processing (RTP) or a cold process [3]
- Commonly traced to deposition and etch effects at the wafer edge [4]
- A full ring points at processes that treat the edge differently by design, and the hardware around the wafer rim [4]

## What engineers usually check

- The edge hardware of the process chamber [4]
- Tool history, and whether the ring appears across several lots; a rising edge-ring rate over weeks can signal slow drift [4]

## Often confused with

- Edge-Loc: a ring that is only partly formed can look like a patch on one section of the edge
- This was the most common WaferLens mistake: 58 Edge-Ring wafers predicted as Edge-Loc, although Edge-Ring is still its best class overall (F1 0.97) [5]