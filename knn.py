import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import missingno as msno
import random

import seaborn as sns

    
def cMSE(y_hat, y, c):
    err = y - y_hat
    loss = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(loss) / len(y)

def cross_val_cMSE(model, X, y, c, cv=5):
    """Perform cross-validation with cMSE as the loss function."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=60)
    cMSE_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        c_train, c_val = c.iloc[train_idx], c.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = cMSE(y_pred, y_val, c_val)
        cMSE_scores.append(score)

    return np.mean(cMSE_scores), np.std(cMSE_scores)

def train_polynomial_regression(X, y, c, degrees, cv=5):
    """Train polynomial regression and select the best degree using cross-validation."""
    best_degree = None
    best_score = float('inf')
    best_std = 0
    scores = []

    for degree in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree)),
            ('regressor', LinearRegression())
        ])
        
        mean_cMSE, std_cMSE = cross_val_cMSE(model, X, y, c, cv=cv)
        scores.append((degree, mean_cMSE, std_cMSE))
        if mean_cMSE < best_score:
            best_score = mean_cMSE
            best_std = std_cMSE
            best_degree = degree

    print("Polynomial Regression Results:")
    print(f"Best Degree: {best_degree}, Mean cMSE: {best_score:.4f}, Std: {best_std:.4f}")
    return best_degree, scores

def train_knn(X, y, c, k_values, cv=5):
    """Train kNN and select the best k using cross-validation."""
    best_k = None
    best_score = float('inf')
    best_std = 0
    scores = []

    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k)
        mean_cMSE, std_cMSE = cross_val_cMSE(model, X, y, c, cv=cv)
        scores.append((k, mean_cMSE, std_cMSE))
        if mean_cMSE < best_score:
            best_score = mean_cMSE
            best_std = std_cMSE
            best_k = k

    print("kNN Results:")
    print(f"Best k: {best_k}, Mean cMSE: {best_score:.4f}, Std: {best_std:.4f}")
    return best_k, scores


train_path = './Datasets/train_data.csv'
test_path = "./Datasets/test_data.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_test_filled = df_test.fillna(0.0)
df_test_filled.drop('Id', axis=1, inplace=True)

df_uncensored = df_train
df_train_clean = df_uncensored.dropna(subset=['SurvivalTime']).copy()

#Fill train
df_train_clean.loc[:, 'GeneticRisk'] = df_train_clean['GeneticRisk'].fillna(df_train_clean['GeneticRisk'].median())
df_train_clean.loc[:, 'ComorbidityIndex'] = df_train_clean['ComorbidityIndex'].fillna(df_train_clean['ComorbidityIndex'].median())
df_train_clean.loc[:, 'TreatmentResponse'] = df_train_clean['TreatmentResponse'].fillna(df_train_clean['TreatmentResponse'].median())
#Fill test
df_test_filled.loc[:, 'GeneticRisk'] = df_test_filled['GeneticRisk'].fillna(df_test_filled['GeneticRisk'].median())
df_test_filled.loc[:, 'ComorbidityIndex'] = df_test_filled['ComorbidityIndex'].fillna(df_test_filled['ComorbidityIndex'].median())
df_test_filled.loc[:, 'TreatmentResponse'] = df_test_filled['TreatmentResponse'].fillna(df_test_filled['TreatmentResponse'].median())

#drop target and useless features
X = df_train_clean.drop(columns=['SurvivalTime', 'Censored', 'Gender', 'GeneticRisk', 'Id'])
c = df_train_clean['Censored']
df_test_filled = df_test_filled.drop(columns=['Gender', 'GeneticRisk'])
y = df_train_clean['SurvivalTime']

# Split the data into train and test sets --------> FEATURES, TARGETS
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2, random_state=random.randint(1, 100))

# Hyperparameter Tuning
degrees = [2, 3, 4, 5, 6]
k_values = [3, 5, 7, 9, 11]

#----------------- Polynomial Model ------------------
best_degree, poly_scores = train_polynomial_regression(X_train, y_train, c_train, degrees)

# Train Final Polynomial Model
poly_model = Pipeline([
    ('poly', PolynomialFeatures(best_degree)),
    ('regressor', LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
y_submission_poly = poly_model.predict(df_test_filled)
#------------------------------------------------------

#----------------- K-Nearest Neighbors -----------------
best_k, knn_scores = train_knn(X_train, y_train, c_train, k_values)

knn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', KNeighborsRegressor(n_neighbors=best_k))  # Ridge regularization
])
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_submission_knn = knn_model.predict(df_test_filled)
#-------------------------------------------------------

# Predictions

# Submission for Polynomial Regression
y_submission_poly = poly_model.predict(df_test_filled)
submission_poly = pd.DataFrame({'Id': range(len(y_submission_poly)), 'Target': y_submission_poly})
submission_poly.to_csv('Nonlinear-submission-poly.csv', index=False)

# Compare Errors Poly
poly_cMSE = cMSE(y_pred_poly, y_test, c_test)
poly_MSE = root_mean_squared_error(y_pred_poly, y_test)
print(f"Best Polynomial cMSE: {poly_cMSE}")
print(f"Best Polynomial MSE: {poly_MSE}")

# Compare Errors KNN
knn_cMSE = cMSE(y_pred_knn, y_test, c_test)
knn_MSE = root_mean_squared_error(y_pred_knn, y_test)
print(f"Best kNN cMSE: {knn_cMSE}")
print(f"Best kNN MSE: {knn_MSE}")

