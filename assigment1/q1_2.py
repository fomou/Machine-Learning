import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from q1_1 import data_matrix_bias, linear_regression_predict, linear_regression_optimize, rmse


# Loading the dataset
df_X = pd.read_csv('Data/X_train.csv')
df_y = pd.read_csv('Data/y_train.csv')
df_X_test = pd.read_csv('Data/X_test.csv')
df_y_test = pd.read_csv('Data/y_test.csv')

# Write your code here:
X_train = df_X.values
y_train = df_y.values
X_test = df_X_test.values
y_test = df_y_test.values
# Find the optimal parameters only using the training set
X_train = data_matrix_bias(X_train)

# fit the model 

w = linear_regression_optimize(y_train, X_train)
X_test_bias = data_matrix_bias(X_test)
y_hat = linear_regression_predict(X_test_bias, w)

print(f'Mean square error on training data is {rmse(y_test,y_hat)} ')
# Report the RMSE and Plot the data on the test set

fig, ax = plt.subplots(1,2)

ax[0].scatter(df_X_test['Experience'],df_y_test['Salary'])
ax[0].plot(y_hat)

ax[1].scatter(df_X_test['Test Score'],df_y_test['Salary'])
plt.tight_layout()
plt.show()

