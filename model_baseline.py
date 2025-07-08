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
from sklearn.model_selection import train_test_split
import missingno as msno
import random

train_path = './Datasets/train_data.csv'
test_path = "./Datasets/test_data.csv"


df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_test_filled = df_test.fillna(0.0)
df_test_filled.drop('Id', axis=1, inplace=True)

df_uncensored = df_train[df_train['Censored'] == 0]

#missing_by_column = df_uncensored.isnull().sum()

df_train_clean = df_uncensored.dropna()

#show graphs
msno.bar(df_train_clean)
msno.dendrogram(df_train_clean)
#plt.show()

print(len(df_train_clean))

missingVal = ['SurvivalTime', 'Censored', 'Id']

# Drop rows with missing target values to use uncensored data
# Assume 'target' is the name of the target variable column

# Separate features and target
X = df_train_clean#.drop(columns=missingVal, axis=1)
y = df_train_clean[['SurvivalTime']]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= random.randint(1, 100))


real_y_test = X_test[['SurvivalTime']]
real_y_censored = X_test[['Censored']]

X_train.drop(columns=missingVal, axis=1, inplace=True)
X_test.drop(columns=missingVal, axis=1, inplace=True)


# Create a pipeline with StandardScaler and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train the baseline model
pipeline.fit(X_train, y_train)

#print(X_test.head())
#print(df_test.head())

# Predict on the test set
y_pred = pipeline.predict(X_test)
y_submission = pipeline.predict(df_test_filled)


#prepare the submission file
def create_submission(submission_np):
    y_submission_df = pd.DataFrame(submission_np, columns=['Target'])
    y_submission_df['id'] = range(0, len(y_submission_df))
    y_submission_df = y_submission_df[['id'] + [col for col in y_submission_df.columns if col != 'id']]
    y_submission_df.to_csv('baseline-submission-00.csv', index=False)



def error_metric(y, y_hat, c):
    err = y-y_hat
    print("########## err ", err)
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]



def cMSE_gradient(y, y_hat, c):
    # Compute the residuals
    err = y - y_hat
    
    # Compute the gradient based on the custom loss
    gradient = -2 * ((1 - c) * err + c * np.maximum(0, err))
    
    return gradient / len(y)  # Average over the number of samples


# Gradient Descent function
def gradient_descent(X, y, c, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.1):
    # Convert y to a NumPy array if it isn't one already
    y = np.array(y).flatten()  # Flatten ensures y is a 1D array, in case it's a pandas Series

    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.randn(n_features)
    bias = 0

    for i in range(n_iterations):
        # Calculate predictions
        y_hat = np.dot(X, weights) + bias

        # Calculate gradient for weights and bias
        grad_weights = np.dot(X.T, cMSE_gradient(y, y_hat, c))
        grad_bias = np.sum(cMSE_gradient(y, y_hat, c))

        # Apply regularization if specified
        if regularization == 'lasso':  # L1 regularization
            grad_weights += lambda_reg * np.sign(weights)
        elif regularization == 'ridge':  # L2 regularization
            grad_weights += lambda_reg * weights

        # Update weights and bias
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        # Optional: Monitor convergence
        if i % 100 == 0:
            loss = np.mean((y - (np.dot(X, weights) + bias))**2)
            print(f"Iteration {i}: Loss = {loss}")
    
    return weights, bias

# Example usage
# X: feature matrix, y: target values, c: coefficient in cMSE loss
weights, bias = gradient_descent(X_train, y_train, c=0.5, learning_rate=0.0001, n_iterations=1000, regularization='ridge', lambda_reg=0.1)

print("weights and bias ", weights, bias)


X_test_scaled = StandardScaler().fit_transform(df_test_filled)  # or use the scaler you used in training

# Calculate predictions using the learned weights and bias
y_submission = np.dot(X_test_scaled, weights) + bias


error = error_metric(real_y_test, y_pred, 0.5)
print(f'Mean Squared Error (cMSE): {error}')

create_submission(y_submission)