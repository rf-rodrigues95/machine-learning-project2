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


# Model 1: HistGradientBoostingRegressor
print("Training HistGradientBoostingRegressor...")
hgb_regressor = HistGradientBoostingRegressor(max_iter=100)
hgb_regressor.fit(X_train, y_train)
hgb_y_pred = hgb_regressor.predict(X_test)

# Metrics
hgb_rmse = root_mean_squared_error(y_test, hgb_y_pred)
hgb_cMSE = cMSE(hgb_y_pred, y_test.values, c_test)
print(f"HistGradientBoostingRegressor RMSE: {hgb_rmse:.4f}")
print(f"HistGradientBoostingRegressor cMSE Loss: {hgb_cMSE:.4f}")

# Plot y vs. y_hat for HistGradientBoostingRegressor
plt.figure()
plt.scatter(y_test, hgb_y_pred, alpha=0.01)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Survival Time (y)")
plt.ylabel("Predicted Survival Time (y_hat)")
plt.title("HistGradientBoostingRegressor: y vs y_hat")
plt.show()

# Model 2: CatBoostRegressor
print("Training CatBoostRegressor...")
catboost_regressor = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=100
)
catboost_regressor.fit(X_train, y_train)
catboost_y_pred = catboost_regressor.predict(X_test)

# Metrics
catboost_rmse = root_mean_squared_error(y_test, catboost_y_pred)
catboost_cMSE = cMSE(catboost_y_pred, y_test.values, c_test)
print(f"CatBoostRegressor RMSE: {catboost_rmse:.4f}")
print(f"CatBoostRegressor cMSE Loss: {catboost_cMSE:.4f}")

# Plot y vs. y_hat for CatBoostRegressor
plt.figure()
plt.scatter(y_test, catboost_y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Survival Time (y)")
plt.ylabel("Predicted Survival Time (y_hat)")
plt.title("CatBoostRegressor: y vs y_hat")
plt.show()

""" best_model = catboost_regressor 
y_submission = best_model.predict(df_test_filled)

# Prepare submission file
submission_df = pd.DataFrame({'id': range(len(y_submission)), 'Target': y_submission})
submission_df.to_csv('model_with_missing_handling_submission.csv', index=False)
print("Submission file saved.") """
