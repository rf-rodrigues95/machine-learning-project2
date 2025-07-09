# Semi-Supervised Learning for Predicting Survival Time in Multiple Myeloma Patients

## Objective

This project aims to predict survival time in multiple myeloma patients using **semi-supervised learning**. We tackle challenges like **missing features**, **right-censored survival times**, and **unlabeled target variables**, all without using traditional survival analysis. Instead, we frame the problem as a **regression task** and use a wide array of machine learning models and techniques learned throughout the course.

---

## Repository Structure

```plaintext
.
├── datasets/                      # Directory with raw and provided datasets
├── cleaned_train.csv             # Preprocessed training data (non-censored, no missing features)
├── cmse_rr/                      # Custom gradient descent training using cMSE loss with regularization
├── knn/                          # Code for K-Nearest Neighbors model
├── model_baseline/              # Baseline Linear Regression (drop censored & missing)
├── model_baseline_rr/           # Improved baseline with imputed missing data
├── task3_2-test/                 # Evaluation of multiple tree-based configs
├── task3_2/                      # Final tree-based model with best configuration
├── Task4.1/                      # Semi-supervised Isomap dimensionality reduction
├── README.md                     # You're here!
