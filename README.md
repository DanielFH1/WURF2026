# POSTECH WURF 2026: Spatial Reasoning in Vision-Language Models

**Researcher:** Seungwoo Lim (Dongguk Univ.)  

**Host Lab:** MLV Lab (Multi-modal Learning & Vision)  

**Advisor:** [Professor Kwang-in Kim](https://scholar.google.com/citations?user=0wIdMGEAAAAJ&hl=en)

**Program:** Winter Undergraduate Research Fellowship (WURF 2026)

This repository contains two distinct research tracks investigating **Allocentric Spatial Reasoning** limitations in modern Vision-Language Models (VLMs).

---

## 📂 Project Overview

### 🎥 [Project 1: Temporal Consistency in Dynamic Scenes](./Project1_MyStudy_SpatialReasoning_Video)
> **Focus:** Fixing the "Flickering" issue in video-based spatial reasoning.

* **Problem:** SOTA VLMs exhibit severe instability (rapid prediction oscillation) when reasoning about spatial relationships in dynamic video streams.
* **Method:** Proposed an **Entropy-guided Adaptive Temporal Smoothing** mechanism.
* **Key Result:** Achieved a **19.0% improvement** in Temporal Consistency Score (TC-Score) by dynamically adjusting smoothing strength ($\alpha$) based on model uncertainty.
* **[👉 Go to Project 1](./Project1_MyStudy_SpatialReasoning_Video)**

### 📊 [Project 2: Data Deep-Dive & Error Analysis](./Project2_datadeepdiveResearch)
> **Focus:** Analyzing *why* models fail in static allocentric tasks.

* **Problem:** A granular analysis of low accuracy (~67%) in spatial benchmarks, identifying major error types like **Horizontal Flips** and **Object Detection Failures**.
* **Hypotheses Tested:**
    1.  **Data Augmentation:** Learning from "Hard Negatives" (Left $\leftrightarrow$ Right flips).
    2.  **Visual Prompting:** Using YOLO-World to apply Bounding Boxes as visual anchors.
    3.  **Chain of Thought (CoT):** Knowledge distillation attempts.
* **Key Result:** Confirmed that Visual Prompting improves specific object-relative tasks but identified "Tunnel Vision" and "Catastrophic Forgetting" trade-offs.
* **[👉 Go to Project 2](./Project2_datadeepdiveResearch)**

---

## 🏗 Repository Structure

```text
.
├── Project1_MyStudy_SpatialReasoning_Video/   # Video stability & Adaptive Alpha experiments
│   ├── run_experiment.py
│   └── ...
├── Project2_datadeepdiveResearch/             # Static image analysis, Augmentation, & Visual Prompting
│   ├── WURF_Data_deep-dive_research.pdf
│   └── ...
└── README.md                                  # You are here