# Machine Learning  
# Predicting Survival Time in Multiple Myeloma Patients (2024)

This project aims to predict survival time in multiple myeloma patients using **semi-supervised learning**. We tackle challenges like **missing features**, **right-censored survival times**, and **unlabeled target variables**, all without using traditional survival analysis. Instead, we frame the problem as a **regression task** and use a wide array of machine learning models and techniques learned throughout the course.

The project is structured into four major tasks, further subdivided as per the assignment specification. All relevant files for each task are clearly indicated.

---

## 📂 Kaggle Competition

**Competition Link**: [ML Nova 2024 – Multiple myeloma survival](https://www.kaggle.com/competitions/machine-learning-nova-multiple-myeloma-survival)

---

## Task 1 – Setting the Baseline

### Task 1.1 – Data Preparation and Validation Pipeline  
📄 **File**: `model_baseline.py`

This step involved extensive data exploration using the `missingno` library. We visualized missing patterns using bar plots, matrix views, heatmaps, and dendrograms to understand the structure of missingness. We observed that dropping all rows with missing features or censored labels drastically reduces the available training data. Pairplots were created for the non-missing subset to understand feature-target relationships.

We defined our feature matrix `X` and target vector `y`, and analyzed two validation strategies:
- Train/validation/test split
- Train/test split with cross-validation

The second approach proved more data-efficient due to limited fully-observed uncensored samples. We defined a **custom cMSE loss** to evaluate performance on censored regression tasks.

---

### Task 1.2 – Learn the Baseline Model  
📄 **File**: `model_baseline.py`

Using only complete, uncensored data, we implemented a **Linear Regression baseline** using a `Pipeline` with `StandardScaler`. The model was trained and validated using the previously established split.

Model evaluation included:
- cMSE score on local validation
- y vs ŷ scatter plots

The resulting predictions were submitted to Kaggle as `baseline-submission-xx.csv`. We also compared local vs Kaggle evaluation to identify any validation mismatch, adjusting our strategy accordingly when needed.

---

### Task 1.3 – Learn with the cMSE  
📄 **File**: `cmse_rr.py`

In this phase, we:
- Computed the gradient of the cMSE loss (handwritten derivation attached in slides).
- Included censored but non-missing data in the training set.
- Implemented **Gradient Descent** for optimizing the cMSE loss.
- Introduced **Ridge and Lasso regularization** to control overfitting.
- Evaluated the learned model via cMSE and y vs ŷ plots.
- Submitted predictions to Kaggle as `cMSE-baseline-submission-xx.csv`.

---

## Task 2 – Nonlinear Models

### Task 2.1 – Model Development  
📄 **File**: `knn.py`

We extended our modeling efforts to **nonlinear models**, focusing on:
- **Polynomial Regression** with varying degrees
- **k-Nearest Neighbors Regression** with hyperparameter tuning (k-selection)

Cross-validation was used to select hyperparameters using the same dataset prepared in Task 1.3, incorporating both censored and uncensored data.

---

### Task 2.2 – Model Evaluation  
📄 **File**: `knn.py`

We compared all nonlinear models to the baseline and cMSE-optimized linear model using:
- cMSE scores
- Error statistics (min, max, mean, std)
- Visual analysis with y vs ŷ plots

The best-performing nonlinear model was submitted to Kaggle as `Nonlinear-submission-xx.csv`.

---

## Task 3 – Handling Missing Data

### Task 3.1 – Missing Data Imputation  
📄 **File**: `task3_1.py`

This task focused on handling missing feature values using supervised data only. We experimented with multiple imputation strategies:
- `SimpleImputer` (mean strategy)
- `KNNImputer`
- `IterativeImputer`

The imputers were fit only on labeled data, and their outputs were passed to our baseline Linear Regression model.

Results were analyzed using:
- cMSE scores
- Comparison plots with and without imputation
- Tables of performance across imputers

The best imputation strategy was then used in downstream models (nonlinear models from Task 2 and tree models in Task 3.2).

---

### Task 3.2 – Models That Handle Missing Data  
📄 **Files**: `task3_2.py`, `task3_2-test.py`

We explored models that **natively handle missing data**, without requiring imputation:
- `HistGradientBoostingRegressor` from scikit-learn
- `CatBoostRegressor` with the **Accelerated Failure Time (AFT)** objective for censored data

Each model was validated using the same cMSE loss and compared against the imputation-based pipelines.

---

### Task 3.3 – Evaluation  
📄 **Files**: `task3_1.py`, `task3_2.py`

We compared all missing-data strategies via:
- y vs ŷ plots
- cMSE scores
- Summary tables (baseline vs imputation vs tree models)

We also combined the best imputation strategy from Task 3.1 with the best tree model from Task 3.2 and evaluated its performance. The top model was submitted to Kaggle as `handle-missing-submission-xx.csv`.

---

## Task 4 – Semi-Supervised Learning for Unlabeled Data

### Task 4.1 – Imputation with Labeled and Unlabeled Data  
📄 **File**: `task4.1.py`

We leveraged the **unlabeled data** (samples with missing SurvivalTime) to improve feature representation. Steps included:
- Re-training the best imputation model from Task 3.1 on both labeled + unlabeled data
- Applying dimensionality reduction using **Isomap**, trained on the full (labeled + unlabeled) feature matrix
- Wrapping the frozen Isomap transformer into a scikit-learn pipeline using `FrozenTransformer`
- Training a Linear Regression model on the transformed features using only the labeled data

---

### Task 4.2 – Evaluation  
📄 **File**: `task4.1.py`

We evaluated the performance of the semi-supervised pipeline by comparing:
- cMSE to baseline and other Task 3 models
- Visual diagnostics via scatter plots
- Summary table of all prior methods

The semi-supervised model's predictions were submitted as `semisupervised-submission-xx.csv`.

---

## 📊 Evaluation Metric

All models were evaluated using the **censored Mean Squared Error (cMSE)**:

```python
def error_metric(y, y_hat, c):
    import numpy as np
    err = y - y_hat
    err = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]

