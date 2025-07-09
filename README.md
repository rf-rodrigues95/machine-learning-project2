# Semi-Supervised Learning for Predicting Survival Time in Multiple Myeloma Patients

## Objective

This project aims to predict survival time in multiple myeloma patients using **semi-supervised learning**. We tackle challenges like **missing features**, **right-censored survival times**, and **unlabeled target variables**, all without using traditional survival analysis. Instead, we frame the problem as a **regression task** and use a wide array of machine learning models and techniques learned throughout the course.

---

## Repository Structure

```plaintext
.
├── datasets/                     # Directory with raw and provided datasets
├── cleaned_train.csv             # Preprocessed training data (no missing features)
├── cmse_rr.py                      # Custom gradient descent training using cMSE loss with regularization
├── knn.py                          # Code for K-Nearest Neighbors model
├── model_baseline.py               # Baseline Linear Regression (drop censored & missing)
├── model_baseline_rr.py            # Improved baseline with imputed missing data
├── task3_1.py                   # Handling missing data using various imputation techniques (Task 3.1)
├── task3_2-test.py                 # Evaluation of multiple tree-based configs (Task 3.2)
├── task3_2.py                      # Final tree-based models with best configuration (Task 3.2)
├── Task4.1.py                      # Semi-supervised Isomap dimensionality reduction (Task 4)
├── Task4.ipynb                      # Semi-supervised Isomap dimensionality reduction (Task 4)
├── README.md                     # You're here!
