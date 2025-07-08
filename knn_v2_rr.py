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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
import missingno as msno
import random

import seaborn as sns

# Fold calculation
def calc_fold(pipeline, X, y, train_ix, test_ix, error_func):
    """Calcular erro de treino e validação para um pipeline."""
    censored = X['Censored']
    X = X.drop(columns=['Censored'])
    
    pipeline.fit(X.iloc[train_ix], y.iloc[train_ix])

    training_err = error_func(pipeline.predict(X.iloc[train_ix]), y.iloc[train_ix], censored.iloc[train_ix])
    validation_err = error_func(pipeline.predict(X.iloc[test_ix]), y.iloc[test_ix], censored.iloc[test_ix])
    
    return training_err, validation_err


def cross_validation(df, pipelines, error_func):
    #print(f"Error function: {error_func.__name__}")

    X = df.drop(columns=['SurvivalTime'])
    y = df['SurvivalTime']
    censored = X['Censored']

    # Separar conjunto de treino e teste inicial
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 100), stratify=censored)

    skf = StratifiedKFold(n_splits=5)
    best_pipeline = None
    best_val_error = float('inf')
    pipeline_errors = {}
    best_degree = -1
    for model_name, model_pipelines in pipelines.items():
        print(f"Modelo: {model_name}")
        pipeline_errors[model_name] = []

        for i, pipeline in enumerate(model_pipelines):
            train_err = val_err = 0
            
            for train_idx, val_idx in skf.split(X_train, X_train['Censored']):
                tr_error, va_error = calc_fold(pipeline, X_train, y_train, train_idx, val_idx, error_func)
                train_err += tr_error
                val_err += va_error
            
            train_err /= skf.get_n_splits()
            val_err /= skf.get_n_splits()

            pipeline_errors[model_name].append((train_err, val_err))
            #print(f"Pipeline {i}: Train Error = {train_err:.4f}, Validation Error = {val_err:.4f}, Degree = {i+1}")
            
            #print(f"Validation error: {pipeline.named_steps['poly'].degree}")

            if 'poly' not in dict(pipeline.named_steps):
                    print(f"Degree = {pipeline.named_steps['regressor'].n_neighbors}, Validation Error = {val_err}")

            if val_err < best_val_error:
                best_val_error = val_err
                best_pipeline = pipeline
                if 'poly' in dict(pipeline.named_steps):
                    best_degree = pipeline.named_steps['poly'].degree

    #print(f"Best Pipeline: {best_pipeline}")
    #if best_degree != -1:
        #print(f"Degree of PolynomialFeatures in best pipeline: {best_degree}")

    return pipeline_errors, best_pipeline

def cMSE(y_hat, y, c):
    err = y - y_hat
    loss = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(loss) / len(y)

train_path = './Datasets/train_data.csv'
test_path = "./Datasets/test_data.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_test_filled = df_test.fillna(0.0)
df_test_filled.drop('Id', axis=1, inplace=True)

df_uncensored = df_train
df_train_clean = df_uncensored.dropna(subset=['SurvivalTime']).copy()
df_train_clean.dropna(inplace=True)

print(len(df_train_clean))

#Fill train
#df_train_clean.loc[:, 'GeneticRisk'] = df_train_clean['GeneticRisk'].fillna(df_train_clean['GeneticRisk'].median())
#df_train_clean.loc[:, 'ComorbidityIndex'] = df_train_clean['ComorbidityIndex'].fillna(df_train_clean['ComorbidityIndex'].median())
#df_train_clean.loc[:, 'TreatmentResponse'] = df_train_clean['TreatmentResponse'].fillna(df_train_clean['TreatmentResponse'].median())
#Fill test
#df_test_filled.loc[:, 'GeneticRisk'] = df_test_filled['GeneticRisk'].fillna(df_test_filled['GeneticRisk'].median())
#df_test_filled.loc[:, 'ComorbidityIndex'] = df_test_filled['ComorbidityIndex'].fillna(df_test_filled['ComorbidityIndex'].median())
#df_test_filled.loc[:, 'TreatmentResponse'] = df_test_filled['TreatmentResponse'].fillna(df_test_filled['TreatmentResponse'].median())

#drop target and useless features
X = df_train_clean.drop(columns=['SurvivalTime', 'Censored', 'Gender', 'GeneticRisk', 'Id'])
c = df_train_clean['Censored']
df_test_filled = df_test_filled.drop(columns=['Gender', 'GeneticRisk'])
y = df_train_clean['SurvivalTime']

def iterate_tests(num_iterations):

    poly_cMSE = 999
    knn_cMSE = 999

    for i in range(num_iterations):

        print("Iteration number ", i)

        # Split the data into train and test sets --------> FEATURES, TARGETS
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2, random_state=random.randint(1, 100))

        # Hyperparameter Tuning
        degrees = [2, 3, 4, 5, 6]
        k_values = [3, 5, 7, 9, 11]

        # Polynomial Regression Cross-Validation
        pipelines_poly = {
            'Polynomial': [
                Pipeline([
                    ('poly', PolynomialFeatures(degree)),
                    ('regressor', LinearRegression())
                ]) for degree in degrees
            ]  
        }
        #print("\n--- Polynomial Regression ---")
        poly_errors, best_poly_pipeline = cross_validation(
        pd.concat([X_train, c_train], axis=1).assign(SurvivalTime=y_train),
        pipelines_poly,
        error_func=cMSE
        )
        best_poly_pipeline.fit(X_train, y_train)
        y_pred_poly = best_poly_pipeline.predict(X_test)

        # kNN Cross-Validation
        pipelines_knn = {
        'kNN': [
            Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', KNeighborsRegressor(n_neighbors=k))
            ]) for k in k_values
        ]
        }
        #print("\n--- kNN Regression ---")
        knn_errors, best_knn_pipeline = cross_validation(
        pd.concat([X_train, c_train], axis=1).assign(SurvivalTime=y_train),
        pipelines_knn,
        error_func=cMSE
        )
        best_knn_pipeline.fit(X_train, y_train)
        y_pred_knn = best_knn_pipeline.predict(X_test)


        # Polynomial Regression Submission
        y_submission_poly_iteration = best_poly_pipeline.predict(df_test_filled)
        #submission_poly.to_csv('Nonlinear-submission-poly.csv', index=False)

        # kNN Submission
        y_submission_knn_iteration = best_knn_pipeline.predict(df_test_filled)

        poly_cMSE_iteration = cMSE(y_pred_poly, y_test, c_test)
        knn_cMSE_iteration = cMSE(y_pred_knn, y_test, c_test)

        if(poly_cMSE > poly_cMSE_iteration):
            poly_cMSE = poly_cMSE_iteration
            poly_MSE = root_mean_squared_error(y_pred_poly, y_test)
            y_submission_poly = y_submission_poly_iteration
            best_degree = best_poly_pipeline.named_steps['poly'].degree
            print(f"Found a better poly {best_degree} | {poly_cMSE}")


        if(knn_cMSE > knn_cMSE_iteration):
            knn_cMSE = knn_cMSE_iteration
            knn_MSE = root_mean_squared_error(y_pred_knn, y_test)
            y_submission_knn = y_submission_knn_iteration
            best_k = best_knn_pipeline.named_steps['regressor'].n_neighbors
            print(f"Found a better knn {best_k} | {knn_cMSE}")


    return y_submission_poly, y_submission_knn, poly_cMSE, knn_cMSE, poly_MSE, knn_MSE, best_degree, best_k




y_submission_poly, y_submission_knn, poly_cMSE, knn_cMSE, poly_MSE, knn_MSE, poly_degree, knn_k = iterate_tests(10)


submission_poly = pd.DataFrame({'id': range(len(y_submission_poly)), 'Target': y_submission_poly})
submission_poly.to_csv('Nonlinear-submission-poly.csv', index=False)

submission_knn = pd.DataFrame({'id': range(len(y_submission_knn)), 'Target': y_submission_knn})
submission_knn.to_csv('Nonlinear-submission-knn.csv', index=False)

# Compare Errors Poly
print(f"Best Polynomial cMSE: {poly_cMSE}")
print(f"Best Polynomial MSE: {poly_MSE}")
print(f"Best Polynomial Degree: {poly_degree}")

# Compare Errors KNN
print(f"Best kNN cMSE: {knn_cMSE}")
print(f"Best kNN MSE: {knn_MSE}")
print(f"Best kNN K: {knn_k}")
