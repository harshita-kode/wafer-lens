---
pattern: none
title: No defect pattern
sources:
  - id: 1
    title: "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets"
    authors: "Ming-Ju Wu, Jyh-Shing R. Jang, Jui-Long Chen"
    year: 2015
    url: "https://doi.org/10.1109/TSM.2014.2364237"
  - id: 2
    title: "Development of a Wafer Defect Pattern Classifier Using Polar Coordinate System Transformed Inputs and Convolutional Neural Networks"
    authors: "Moo Hyun Kim, Tae Seon Kim"
    year: 2024
    url: "https://doi.org/10.3390/electronics13071360"
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

- No systematic cluster, ring or line of failed dies
- A few scattered failures are normal and spread fairly evenly, like salt noise on an image [2]
- Most labelled wafers in WM-811K fall in this class: 147,431 of 172,950 [1][2]

## Common causes

- The scattered background failures come from particles and the general process environment [2][4]
- These are hard to remove directly and usually need gradual environment or equipment improvements [2]

## What engineers usually check

- No targeted investigation is usually needed for a single wafer
- Watch the background failure rate over time; a rising rate can signal a gradual problem before it becomes a clear pattern [4]

## Often confused with

- Small or faint patterns can blend into the background: in WaferLens testing, 36 Edge-Loc, 31 Loc and 29 Scratch wafers were predicted as none [5]
- Random: a random pattern is much denser than normal background noise