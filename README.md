# Hierarchical scRNA-seq Classification Pipeline

A hierarchical machine learning framework for the classification of **Cell Types** and **Disease Status** in single-cell RNA sequencing (scRNA-seq) data from cancer microenvironments.

## 🚀 Performance Overview
* **Cell Type Accuracy:** 100% (Cancer vs. T-Cell vs. Fibroblast)
* **Disease Status Accuracy:** 87.5% (Tumor vs. Healthy Control)
* **Key Finding:** Weak learners (Decision Stumps) + AdaBoost outperformed complex SVMs by **~16%**.

## 🛠 Project Architecture
This repository implements a **two-layer hierarchical pipeline**:

1. **Layer 1 (The Ensemble):** Uses Majority Voting across five distinct learners to categorize cells into broad biological types.
2. **Layer 2 (The Booster):** Employs an AdaBoost framework to refine predictions for non-cancer cells, distinguishing between tumor-associated and healthy populations.

## 🔬 Methodology
* **Boosting:** Utilized AdaBoost to iteratively focus on "hard-to-classify" samples, which is critical for the subtle transcriptional differences in disease states.
* **Imbalance Handling:** Integrated strategic sampling and ensemble diversity to mitigate the 76% majority-class bias (Tumor-associated cells).
* **Data Source:** Synthetic scRNA-seq tumor microenvironment data modeling 3,000 cells across 5 key gene markers and inflammatory pathway scores.

## 📊 Results & Visualization
The pipeline's effectiveness is validated through a comparison of **weak learners vs. complex models**, demonstrating that for high-dimensional biological data, ensemble boosting on simple models provides superior generalization.
