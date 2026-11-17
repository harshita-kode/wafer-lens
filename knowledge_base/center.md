---
pattern: Center
title: Center defect pattern
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

- A group of failed dies concentrated in the middle of the wafer [2][3]
- Its position is consistent from wafer to wafer, which makes it one of the easier patterns to recognise [2]

## Common causes

- Often associated with irregular RF operation or unusual liquid flow during processing [3]
- Commonly traced to dispense and spin dynamics, or to effects at the centre of a process chamber [4]
- Center and donut patterns tend to point at processes acting on a spinning or centre-fed wafer [4]

## What engineers usually check

- The coat and develop modules are a common first check [4]
- Confirm the suspicion against the history of the tools the wafer passed through [4]
- Check whether the pattern repeats across several wafers or lots, since one map suggests a cause but many maps confirm it [4]

## Often confused with

- Donut: both are centred on the wafer, but a donut has good dies in the very middle
- WaferLens classifies Center well (F1 0.94), with no single dominant confusion [5]