---
pattern: general
title: Wafer maps, testing and yield
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
---

## What a wafer map is

- A wafer holds tens to hundreds of chips (dies), built together through hundreds of chemical, optical and mechanical processing steps [2]
- After testing, each die gets a result code, and plotting these codes by position gives a wafer bin map [2]
- In the simplest form, each die is shown as pass or fail [2]
- WM-811K, the dataset used in WaferLens, contains 811,457 wafer maps from real production; 172,950 were labelled by domain experts, and 147,431 of those have no defect pattern [1][2]

## How wafers are tested

- A common wafer-level test is electrical die sorting (EDS), which checks each chip's electrical characteristics with AC and DC parameter tests [2]
- Each die is given a bin code based on its test result, which helps identify the kind of failure [2]

## What yield means

- Yield is the share of dies on a wafer that pass testing
- EDS results provide information on fabrication yield [2]
- Wafer-level defect screening is central to yield and cost, especially for products where many dies are stacked, since one bad die can ruin a whole package [3]

## Why spatial patterns matter

- Patterned failures are mainly driven by controllable factors such as process parameters, equipment faults and improper operation [2]
- Scattered random failures are usually linked to the process environment and are harder to control directly [2]
- Each pattern's shape reflects the geometry of the fault that caused it, so the pattern narrows down which tools or steps to investigate [4]
- Classifying the pattern correctly is a prerequisite for correct root-cause diagnosis; a wrong label can send engineers to the wrong process module [3]
- Many real wafers contain more than one pattern at once, even though WM-811K gives each wafer a single label [2]