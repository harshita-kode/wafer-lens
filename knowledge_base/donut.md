---
pattern: Donut
title: Donut defect pattern
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

- A ring of failed dies with an opening in the middle [3]
- Few failures at the very centre and at the outer edge [2]
- The ring sits between the centre and the edge [4]

## Common causes

- Often linked to residue that resists removal during photoresist cleaning [3]
- Commonly traced to photoresist redeposition and spin dynamics [4]

## What engineers usually check

- The coat module and the photoresist batch used [4]
- Tool history for the wafers showing the pattern, and whether it repeats across lots [4]

## Often confused with

- Edge-Ring: both are rings, but a donut ring sits inside the wafer, not on the rim
- Center: both are centred, but a center pattern has no hole in the middle
- WaferLens finds almost all real donuts (recall 0.93) but over-predicts the class (precision 0.73), so some donut predictions are false alarms [5]