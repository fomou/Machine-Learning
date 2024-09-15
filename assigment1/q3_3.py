import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from q1_1 import rmse
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import gradient_descent_regression
from q1_1 import linear_regression_predict

# Load the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values


np.random.seed(42)  # For reproducibility
n_features = X_train.shape[1]
initial_w = np.random.normal(0, 1, size=n_features)
initial_b = 0.0


learning_rate = 1e-8  # You can change this value to get better results
num_epochs = 1000
ridge_hyperparameter = 0.1 # You can change this value to get better results

w, b = gradient_descent_regression(X_train,y_train,num_epochs=300,learning_rate=learning_rate)

print(w)
print(b)
# Provide your code here ...