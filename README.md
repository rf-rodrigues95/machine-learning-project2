# Machine Learning
**NOVA School of Science and Technology (NOVA FCT) – 2024/2025**  
**Final Grade:** 17.5   
**Author:** 
  - Ricardo Rodrigues (rf-rodrigues95)
  - Guilherme Antunes (gpantunes)

# Predicting Survival Time in Multiple Myeloma Patients (2024)

This project aims to predict survival time in multiple myeloma patients using **semi-supervised learning**. We tackle challenges like **missing features**, **right-censored survival times**, and **unlabeled target variables**, all without using traditional survival analysis. Instead, we frame the problem as a **regression task** and use a wide array of machine learning models and techniques learned throughout the course.

The project is structured into four major tasks, further subdivided as per the assignment specification. All relevant files for each task are clearly indicated.

---

## 📂 Kaggle Competition

**Competition Link**: [ML Nova 2024 – Multiple myeloma survival](https://www.kaggle.com/competitions/machine-learning-nova-multiple-myeloma-survival)

## Task 1 – Setting the Baseline  
### Task 1.1 – Data Preparation and Validation Pipeline  
📄 **File**: `model_baseline.py`

- Used missingno to visualize missing values in the dataset with bar plots, heatmaps, matrices, and dendrograms.
- Dropped columns containing features with missing data and all censored or unlabeled data points.
- Counted the number of remaining examples after filtering.
- Generated pairplots between the remaining features and the survival time target.
- Defined the feature matrix X and target vector y.
- Compared train/validation/test splits with cross-validation to determine the most data-efficient validation procedure.
- Implemented the censored Mean Squared Error (cMSE) as the evaluation metric.

### Task 1.2 – Learn the Baseline Model  
📄 **File**: `model_baseline.py`

- Trained a baseline Linear Regression model on the uncensored, non-missing data.
- Created a pipeline that includes a StandardScaler followed by the linear regressor.
- Computed the cMSE and plotted the y versus ŷ predictions.

Submitted the baseline predictions to Kaggle in the specified format the format baseline-submission-xx.csv.  

### Task 1.3 – Learn with the cMSE  
📄 **File**: `model_baseline.py`, `cmse.py`     
**Note**: The file `cmse.py` was created to independently implement and test the cMSE loss function before integrating it into the baseline model pipeline.

- Derived the analytical gradient of the cMSE loss function.
- Included censored data (with known values) in training.
- Implemented Gradient Descent manually using the derived gradient expression.
- Added optional Lasso and Ridge regularization to the Gradient Descent implementation.

Submitted predictions trained with cMSE to Kaggle under the format cMSE-baseline-submission-xx.csv.

---

## Task 2 – Nonlinear Models  
### Task 2.1 – Development  
📄 **File**: `knn.py`

- Implemented functions for training Polynomial and k-Nearest Neighbors on the data prepared in Task 1.1. using the validation procedure determined in Task 1.1 and Task 1.2. Used the same data from task 1.3.
- Selectd the model hyperparamenters, like the polynomial degree and the $k$ using cross validation for model selection.  

### Task 2.2 – Evaluation  
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

- Used Scikit-Learn's imputation techniques to complete missing data in the features.
- Trained the baseline Linear Regression model on the imputed dataset.
- Compared imputation strategies and selected the best one for reuse in later tasks.
- Applied the best imputation strategy to nonlinear models from Task 2.  

### Task 3.2 – Models That Handle Missing Data  
📄 **Files**: `task3_2.py`, `task3_2-test.py`  
**Note**: File `task3_2-test.py` evaluates multiple tree-based configs, the best are selected to be used in `task3_2.py`

Implemented models that natively handle missing values:
- HistGradientBoostingRegressor from Scikit-Learn
- CatBoostRegressor using the Accelerated Failure Time (AFT) loss
 
Integrated CatBoost-specific survival regression setup using their official tutorial as reference.  

### Task 3.3 – Evaluation  
📄 **Files**: `task3_1.py`, `task3_2.py`

- Compared the models from Tasks 3.1 and 3.2 with the baseline.
- Applied the best imputation strategy to the best tree-based model and reran training.
- Submitted the best-performing predictions to Kaggle as handle-missing-submission-xx.csv.
    
---

## Task 4 – Semi-Supervised Learning for Unlabeled Data
### Task 4.1 – Imputation with Labeled and Unlabeled Data  
📄 **File**: `task4.1.py`

- Refit the best imputers from Task 3.1 using both labeled and unlabeled features.
- Trained a Linear Regression model on the labeled portion of the imputed dataset.
- Trained an Isomap model for dimensionality reduction using both labeled and unlabeled examples.
- Wrapped the fitted Isomap model using a custom FrozenTransformer class to allow integration into a Scikit-Learn pipeline.
- Constructed a pipeline consisting of imputation, scaling, Isomap, and Linear Regression for final training.  

### Task 4.2 – Evaluation  
📄 **File**: `task4.1.py`

- Compared the semi-supervised models with the baseline.
- Reused the best imputation method from Task 3.1 with the best model from Task 3.2.
- Submitted final semi-supervised predictions to Kaggle as semisupervised-submission-xx.csv.

---

## 📊 Evaluation Metric

All models were evaluated using the **censored Mean Squared Error (cMSE)**:

```python
def error_metric(y, y_hat, c):
    import numpy as np
    err = y - y_hat
    err = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]

