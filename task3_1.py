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

def cross_val_cMSE(pipeline, X, y, c, cv=5):
    scores = []
    fold_size = len(X) // cv

    for i in range(cv):
        X_val = X[i * fold_size:(i + 1) * fold_size]
        y_val = y[i * fold_size:(i + 1) * fold_size]
        c_val = c[i * fold_size:(i + 1) * fold_size]

        X_train = np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]))
        y_train = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]))

        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        scores.append(cMSE(y_val_pred, y_val, c_val))

    return scores

def cMSE(y_hat, y, c):
    err = y - y_hat
    loss = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(loss) / len(y)

def apply_imputation(df, column,  strategy='mean'):
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    elif strategy == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif strategy == 'iterative':
        imputer = IterativeImputer(max_iter=10, random_state=60)

    print(df[column])
    df[column] = imputer.fit_transform(df[[column]])
    print(df[column])
    return df

train_path = './Datasets/train_data.csv'
test_path = "./Datasets/test_data.csv"

df_train_clean = pd.read_csv(train_path)
df_test_filled = pd.read_csv(test_path)

#['mean', 'median', 'most_frequent', 'knn', 'iterative']
imputation_strategies = ['iterative']
for strategy in imputation_strategies:
    print(f"Applying {strategy} imputation strategy...")
    
    # Apply imputation to columns in both train and test datasets
    df_train_clean = apply_imputation(df_train_clean, 'GeneticRisk', strategy)
    df_train_clean = apply_imputation(df_train_clean, 'ComorbidityIndex', strategy)
    df_train_clean = apply_imputation(df_train_clean, 'TreatmentResponse', strategy)

    df_test_filled = apply_imputation(df_test_filled, 'GeneticRisk', strategy)
    df_test_filled = apply_imputation(df_test_filled, 'ComorbidityIndex', strategy)
    df_test_filled = apply_imputation(df_test_filled, 'TreatmentResponse', strategy)


df_test_filled.drop('Id', axis=1, inplace=True)
df_uncensored = df_train_clean[df_train_clean['Censored'] == 0]
df_train_clean = df_uncensored.dropna(subset=['SurvivalTime']).copy()
X = df_train_clean.drop(columns=['SurvivalTime', 'Censored', 'Gender', 'GeneticRisk', 'Id'])
df_test_filled = df_test_filled.drop(columns=['Gender', 'GeneticRisk'])
y = df_train_clean['SurvivalTime']
c = df_train_clean['Censored'].values

# Split the data into train and test sets --------> FEATURES, TARGETS
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2, random_state= 60)

# Create a pipeline with StandardScaler and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())  # Ridge regularization
])

# Cross-validated RMSE
cv_scores = cross_val_cMSE(pipeline, X_train, y_train, c_train, cv=5)
print(f"Cross-validated RMSE: {np.sqrt(np.mean(cv_scores)):.4f}")

# Baseline model training
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_submission = pipeline.predict(df_test_filled)


# Preparing Submission File
y_submission_df = pd.DataFrame(y_submission, columns=['Target'])
y_submission_df['id'] = range(0, len(y_submission_df))
y_submission_df = y_submission_df[['id'] + [col for col in y_submission_df.columns if col != 'id']]
y_submission_df.to_csv('cMSE-baseline-submission-00.csv', index=False)

error = root_mean_squared_error(y_pred, y_test)

print(f'Mean Squared Error (MSE): {error}')

test_cMSE = cMSE(y_pred, y_test.values, c_test)
print(f"Test cMSE Loss: {test_cMSE:.4f}")
