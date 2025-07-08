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
from sklearn.model_selection import cross_val_score, train_test_split
import missingno as msno
import seaborn as sns
import random

import seaborn as sns

train_path = './Datasets/train_data.csv'
test_path = "./Datasets/test_data.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

sns.pairplot(df_train)
#plt.show()

df_test_filled = df_test.fillna(0.0)
#df_test_filled.drop(columns=['Id', 'Gender', 'GeneticRisk'], axis=1, inplace=True)
df_test_filled.drop(columns=['Id'], axis=1, inplace=True)


df_uncensored = df_train[df_train['Censored'] == 0]
#df_train_clean = df_uncensored.dropna(subset=['SurvivalTime'])
df_train_clean = df_uncensored.dropna()

# Impute missing values in columns with missing data
#df_train_clean['GeneticRisk'].fillna(df_train_clean['GeneticRisk'].median(), inplace=True)
#df_train_clean['ComorbidityIndex'].fillna(df_train_clean['ComorbidityIndex'].median(), inplace=True)
#df_train_clean['TreatmentResponse'].fillna(df_train_clean['TreatmentResponse'].median(), inplace=True)

df_train_clean.to_csv('Cleaned_train.csv', index=False)

print(f'Number of rows after cleaning: {len(df_train_clean)}')


#X = df_train_clean.drop(columns=['SurvivalTime', 'Censored', 'Gender', 'GeneticRisk', 'Id'])
X = df_train_clean.drop(columns=['SurvivalTime', 'Censored', 'Id'])

y = df_train_clean['SurvivalTime']

sns.pairplot(df_train_clean, x_vars=X.columns, y_vars='SurvivalTime', height=2.5)
#plt.show()

# Split the data into train and test sets --------> FEATURES, TARGETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 60)

# Create a pipeline with StandardScaler and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validated RMSE: {np.sqrt(-np.mean(cross_val_scores))}')

# Train the baseline model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)


y_submission = pipeline.predict(df_test_filled)


#prepare the submission file
y_submission_df = pd.DataFrame(y_submission, columns=['Target'])
y_submission_df['id'] = range(0, len(y_submission_df))
y_submission_df = y_submission_df[['id'] + [col for col in y_submission_df.columns if col != 'id']]
y_submission_df.to_csv('baseline-submission-00.csv', index=False)



def error_metric(y, y_hat, c):
    err = y-y_hat
    print("########## err ", err)
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]



#nan_array = df_train_clean.isna().astype(int).values

# Compute the MSE (equivalent to cMSE for uncensored data)
error = root_mean_squared_error(y_pred, y_test)
#error = error_metric(real_y_test, y_pred, real_y_censored)

print(f'Mean Squared Error (MSE): {error}')