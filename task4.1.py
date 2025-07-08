import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.manifold import Isomap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import random
import missingno as msno
import warnings
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")


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
    loss = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(loss) / len(y)


class FrozenTransformer(BaseEstimator):
    def __init__(self, fitted_transformer):
        self.fitted_transformer = fitted_transformer

    def __getattr__(self, name):
        # `fitted_transformer`'s attributes are now accessible
        return getattr(self.fitted_transformer, name)

    def __sklearn_clone__(self):
        return self

    def fit(self, X, y=None):
        # Fitting does not change the state of the estimator
        return self

    def transform(self, X, y=None):
        # transform only transforms the data
        return self.fitted_transformer.transform(X)

    def fit_transform(self, X, y=None):
        # fit_transform only transforms the data
        return self.fitted_transformer.transform(X)


def apply_imputation(df, column, strategy='mean'):
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

    # print(df[column])
    df[column] = imputer.fit_transform(df[[column]])
    # print(df[column])
    return df


def learn_and_apply_imputer(data_features_with_labels, data_without_labels, strategy='iterative'):
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    elif strategy == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif strategy == 'iterative':
        imputer = IterativeImputer(max_iter=10, random_state=random.randint(0,
                                                                            100))  # TODO: aqui se calhar temos de mudar o random state para randint(0,100)

    imputer.fit(data_features_with_labels)
    data_without_labels_imputed = imputer.transform(data_without_labels)

    return pd.DataFrame(data_without_labels_imputed, columns=data_without_labels.columns)


def process_data():
    train_path = './Datasets/train_data.csv'
    test_path = "./Datasets/test_data.csv"

    df_train_clean = pd.read_csv(train_path)
    df_test_filled = pd.read_csv(test_path)

    data_features_with_labels = df_train_clean.dropna(subset=['SurvivalTime'])
    data_without_labels = df_train_clean[df_train_clean['SurvivalTime'].isnull()]

    data_features_with_labels = apply_imputation(data_features_with_labels, 'GeneticRisk', strategy='iterative')
    data_features_with_labels = apply_imputation(data_features_with_labels, 'ComorbidityIndex', strategy='iterative')
    data_features_with_labels = apply_imputation(data_features_with_labels, 'TreatmentResponse', strategy='iterative')

    df_test_filled = apply_imputation(df_test_filled, 'GeneticRisk', strategy='iterative')
    df_test_filled = apply_imputation(df_test_filled, 'ComorbidityIndex', strategy='iterative')
    df_test_filled = apply_imputation(df_test_filled, 'TreatmentResponse', strategy='iterative')

    # É usar os dados sem label com as features dos dados com label para aprender um imputer e/ou um isomap/pca
    data_without_labels = learn_and_apply_imputer(data_features_with_labels, data_without_labels, strategy='iterative')
    # msno.bar(data_without_labels)

    full_data = pd.concat([data_features_with_labels, data_without_labels])
    # msno.bar(full_data)

    df_test_filled.drop('Id', axis=1, inplace=True)

    # experimenta com o censored e sem censored
    df_uncensored = full_data[full_data['Censored'] == 0]
    # msno.bar(df_uncensored)

    X = full_data.drop(columns=['SurvivalTime', 'Censored', 'Gender', 'GeneticRisk', 'Id'])
    df_test_filled = df_test_filled.drop(columns=['Gender', 'GeneticRisk'])

    y = full_data['SurvivalTime']
    c = full_data['Censored'].values

    # Split the data into train and test sets --------> FEATURES, TARGETS
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2,
                                                                         random_state=random.randint(0, 100))

    return X_train, X_test, y_train, y_test, c_train, c_test, df_test_filled


def reduce_dimensions(X_train):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    pca = PCA(n_components=5)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    # data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])

    return data_pca


def run_model():
    X_train, X_test, y_train, y_test, c_train, c_test, df_test_filled = process_data()
    # X_train = reduce_dimensions(X_train)

    iso = Isomap(n_components=5)
    iso.fit(X_train)

    model = make_pipeline(
        StandardScaler(),
        FrozenTransformer(iso).fitted_transformer,
        LinearRegression())

    # Cross-validated RMSE
    cv_scores = cross_val_cMSE(model, X_train, y_train, c_train, cv=5)
    cv_score = np.sqrt(np.mean(cv_scores))

    # Baseline model training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_submission = model.predict(df_test_filled)

    # Preparing Submission File
    y_submission_df = pd.DataFrame(y_submission, columns=['Target'])
    y_submission_df['id'] = range(0, len(y_submission_df))
    y_submission_df = y_submission_df[['id'] + [col for col in y_submission_df.columns if col != 'id']]
    # y_submission_df.to_csv('cMSE-baseline-submission-00.csv', index=False)

    error = root_mean_squared_error(y_pred, y_test)
    test_cMSE = cMSE(y_pred, y_test.values, c_test)

    return cv_score, error, test_cMSE, y_submission_df


def find_best_result(results):
    best_model_run = None

    for test_cMSE, result, predictions in results:
        if best_model_run is None or test_cMSE < best_model_run[0]:
            best_model_run = (test_cMSE, result, predictions)

    return best_model_run


def run_iterations():
    iterations = 100
    results = []
    min_test_cMSE = 1000000

    for i in range(iterations):
        cv_score, error, test_cMSE, y_submission = run_model()

        if test_cMSE < min_test_cMSE:
            min_test_cMSE = test_cMSE
            y_submission_df = pd.DataFrame(y_submission, columns=['Target'])
            y_submission_df['id'] = range(0, len(y_submission_df))
            y_submission_df = y_submission_df[['id'] + [col for col in y_submission_df.columns if col != 'id']]
            y_submission_df.to_csv('cMSE-task4-submission-00.csv', index=False)

        result = (
            f"Cross-validated RMSE: {cv_score:.4f}\n"
            f"Mean Squared Error (MSE): {error:.6f}\n"
            f"Test cMSE Loss: {test_cMSE:.4f}\n"
        )

        results.append((test_cMSE, result, y_submission))

    best_result = find_best_result(results)
    print(best_result[1])
    best_result[2].to_csv('cMSE-task4-submission-00.csv', index=False)


run_iterations()
