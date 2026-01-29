# Data Deep-Dive Research: Analyzing Spatial Reasoning Failures in VLMs

**Researcher:** Seungwoo Lim (Dongguk Univ.) 

**Program:** POSTECH WURF 2026 (Winter Undergraduate Research Fellowship) 

**Host Lab:** MLV Lab (Multi-modal Learning & Vision) 

**Advisor:** [Professor Kwang-in Kim](https://scholar.google.com/citations?user=0wIdMGEAAAAJ&hl=en)

**Presentation** : [Presentation](./WURF_Data_deep-dive_research.pdf)

**Base Model:** Qwen2.5-VL (7B & 3B)
 
---

## 1. Motivation & Problem Statement
Following the preliminary findings on Vision-Language Models (VLMs), we identified that models suffer from low accuracy (approx. 67%) in spatial reasoning tasks.
Most importantly, it was unclear *why* these models fail—whether it is due to simple direction confusion (Left $\leftrightarrow$ Right) or a fundamental lack of allocentric understanding.
This research conducts a **"Deep-Dive"** into the data and model behaviors to analyze error patterns and validates three hypotheses to improve spatial reasoning performance.
We aim to solve issues like **Horizontal Flips** and **Object Detection Failures** through targeted experiments.

---

## 2. Methodology: Three Hypotheses
We investigated three distinct approaches to overcome the limitations of the baseline (Vanilla MVSM).

### Hypothesis 1: Data Augmentation (Hard Negatives)
Inspired by ALBEF (NeurIPS 2021), we focused on learning "hard negatives"—data that is semantically similar but differs in fine-grained details.
1.  **Augmentation:** Expanded the training dataset (4548 $\to$ 9048 samples) by flipping images and corresponding labels (Left $\leftrightarrow$ Right).
2.  **Goal:** To reduce **Horizontal Flip Errors**, which accounted for 41.5% of total errors.

### Hypothesis 2: Bounding Box + Visual Prompting
The model often fails to recognize the correct reference object ("Tunnel Vision" or wrong focus).
1.  **Reference Point Detection:** Utilized **Ultralytics YOLO-World** to detect objects specified in the text.
2.  **Visual Prompting:** Drew Red Bounding Boxes around reference objects to "guide" the model's attention.
3.  **Saliency Map Analysis:** Calculated gradients $S_{i}=|\frac{\partial f(x)}{\partial x_{i}}|$ to visualize if the model focuses on the correct region.

### Hypothesis 3: Chain of Thought (CoT) Distillation
Attempted to improve reasoning logic through knowledge distillation.
* **Teacher:** Qwen2.5-7B (Constructs step-by-step logic).
* **Student:** Qwen2.5-3B (Learns to mimic the reasoning process).
* **Process:** "Let's think step by step to determine the spatial relationship..."

---

## 3. Repository Structure & Key Scripts

This repository contains scripts for data augmentation, visual prompting, and performance evaluation.

### 🛠️ Core Experiments
| File | Description |
| :--- | :--- |
| **`run_data_augmentation.py`** | **[Hypothesis 1]** Generates augmented training data by applying horizontal flips and label inversion to create "Hard Negatives". |
| **`run_visual_prompt.py`** | **[Hypothesis 2]** Integrates **YOLO-World** to detect reference objects and overlays Bounding Boxes on input images for inference. |
| **`run_cot_distillation.py`** | **[Hypothesis 3]** Performs knowledge distillation from the 7B Teacher model to the 3B Student model using Chain-of-Thought prompts. |
| **`analyze_error_types.py`** | **[Analysis]** Classifies failure modes into Horizontal Flip, Orthogonal Error, Vertical Flip, etc. (as shown in the Presentation). |

### 📊 Visualization & Analysis
| File | Description |
| :--- | :--- |
| **`visualize_saliency.py`** | Generates **Saliency Maps** to compare model focus between original images and bounding-box prompted images. |
| **`plot_benchmark_comparison.py`** | Plots comparative bar charts (Baseline vs. Augmented vs. Visual Prompt) for task-wise accuracy. |

---

## 4. Key Results & Analysis

### A. Granular Error Analysis
We broke down the failure modes of the baseline model.
* **Horizontal Flip (Left $\leftrightarrow$ Right):** 41.5% (Most dominant error).
* **Orthogonal Error (Front $\leftrightarrow$ Left):** 31.5%.
* **Vertical Flip (Top $\leftrightarrow$ Bottom):** 24.6%.

### B. Hypothesis 1: Data Augmentation
While augmentation improved the targeted **Relative Direction** task, it led to a performance drop in other tasks.
* **Camera Rel Dir:** +3.9% improvement.
* **Observation:** The model suffered from **Catastrophic Forgetting**, losing general scene understanding capabilities.

### C. Hypothesis 2: Visual Prompting (Bounding Box)
Using Bounding Boxes helped anchor the model's perspective for specific objects but induced a "Tunnel Vision" effect.
| Method | Camera Rel Dir | Person Scene Sim | Total Accuracy |
| :--- | :--- | :--- | :--- |
| **Baseline** | 59.4% | **67.0%** | **67.7%** |
| **Visual Prompt (Ours)** | 58.9% | 56.9% | 65.9% |

* **Insight:** Explicit bounding boxes improved specific object-to-object relations (+2.4% in Person Rel Dir) but degraded performance in holistic scene simulation tasks (-10.1%).

### D. Hypothesis 3: Chain of Thought (CoT)
The CoT experiment resulted in significant performance degradation (Total Accuracy: 49.1%).
* **Cause:** The 7B teacher model was not sufficient to generate high-quality reasoning paths, introducing **noise** rather than logic into the student model.

---

## 5. Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- `ultralytics` (for YOLO-World)

### Installation
```bash
git clone [https://github.com/seungwoo-lim/WURF2026-DataDeepDive.git](https://github.com/seungwoo-lim/WURF2026-DataDeepDive.git)
cd WURF2026-DataDeepDive
pip install -r requirements.txt
```

## Acknowledgements
This project is part of the 2026 POSTECH WURF program.
Special thanks to the MLV Lab for supporting the computing resources and guiding me.