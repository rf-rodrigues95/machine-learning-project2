import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ParameterGrid
import missingno as msno
import random

import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def evaluate_model(model, params, X_train, y_train, c_train, X_test, y_test, c_test):
    """
    Train and evaluate a model with given parameters.
    """
    print(f"Evaluating {model.__class__.__name__} with parameters: {params}")
    start_time = time.time()

    model.set_params(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    cmse_loss = cMSE(y_pred, y_test.values, c_test)
    elapsed_time = time.time() - start_time
    
    print(f"RMSE: {rmse:.4f}, cMSE: {cmse_loss:.4f}, Time: {elapsed_time:.2f}s\n")
    return cmse_loss, rmse, params


def cMSE(y_hat, y, c):
    err = y - y_hat
    loss = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(loss) / len(y)

train_path = './Datasets/train_data.csv'
test_path = "./Datasets/test_data.csv"

df_train_clean = pd.read_csv(train_path)
df_test_filled = pd.read_csv(test_path)


df_test_filled.drop('Id', axis=1, inplace=True)
df_test_filled = df_test_filled.drop(columns=['Gender', 'GeneticRisk'])

df_uncensored = df_train_clean[df_train_clean['Censored'] == 0]
df_train_clean = df_uncensored.dropna(subset=['SurvivalTime']).copy()

X = df_train_clean.drop(columns=['SurvivalTime', 'Censored', 'Gender', 'GeneticRisk', 'Id'])
y = df_train_clean['SurvivalTime']
c = df_train_clean['Censored'].values

# Split the data into train and test sets --------> FEATURES, TARGETS
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2, random_state= 60)


hgb_param_grid = ParameterGrid({
    'max_iter': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [20, 30, 50]
})

catboost_param_grid = ParameterGrid({
    'iterations': [300, 500, 700],
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [3, 5, 7]
})

# Initialize models
hgb_model = HistGradientBoostingRegressor()

# Experiment with both models
best_results = {'model': None, 'params': None, 'cMSE': float('inf'), 'RMSE': None}
best_hgb = {'params': None, 'cMSE': float('inf'), 'RMSE': None}

print("Starting experiments for HistGradientBoostingRegressor...")
for params in hgb_param_grid:
    cmse_loss, rmse, params = evaluate_model(hgb_model, params, X_train, y_train, c_train, X_test, y_test, c_test)
    if cmse_loss < best_results['cMSE']:
        best_hgb.update({'params': params, 'cMSE': cmse_loss, 'RMSE': rmse})
        best_results.update({'model': 'HistGradientBoostingRegressor', 'params': params, 'cMSE': cmse_loss, 'RMSE': rmse})

print("\nBest HGB :")
print(f"Parameters: {best_hgb['params']} ; CMSE: {best_hgb['cMSE']:.4f} ; RMSE: {best_hgb['RMSE']:.4f}")

best_cat = {'params': None, 'cMSE': float('inf'), 'RMSE': None}
print("Starting experiments for CatBoostRegressor...")
for params in catboost_param_grid:
    catboost_model = CatBoostRegressor(verbose=0, loss_function='RMSE', **params)
    cmse_loss, rmse, params = evaluate_model(catboost_model, params, X_train, y_train, c_train, X_test, y_test, c_test)
    if cmse_loss < best_results['cMSE']:
        best_cat.update({'params': params, 'cMSE': cmse_loss, 'RMSE': rmse})
        best_results.update({'model': 'CatBoostRegressor', 'params': params, 'cMSE': cmse_loss, 'RMSE': rmse})

print("\nBest CAT :")
print(f"Parameters: {best_cat['params']} ; CMSE: {best_cat['cMSE']:.4f} ; RMSE: {best_cat['RMSE']:.4f}")

# Display the best model and parameters
print("\nBest Model Configuration:")
print(f"Model: {best_results['model']}")
print(f"Parameters: {best_results['params']}")
print(f"Lowest cMSE: {best_results['cMSE']:.4f}")
print(f"Corresponding RMSE: {best_results['RMSE']:.4f}")

""" # Train and prepare the best model for submission
if best_results['model'] == 'HistGradientBoostingRegressor':
    best_model = HistGradientBoostingRegressor(**best_results['params'])
else:
    best_model = CatBoostRegressor(verbose=0, loss_function='RMSE', **best_results['params'])

best_model.fit(X_train, y_train)
y_submission = best_model.predict(df_test_filled)

# Prepare submission file
submission_df = pd.DataFrame({'id': range(len(y_submission)), 'Target': y_submission})
submission_df.to_csv('best_model_submission.csv', index=False)
print("Submission file saved.") """
