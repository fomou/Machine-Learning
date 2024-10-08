import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from q2_2 import cross_validation_linear_regression

# Define a range of alpha values for hyperparameter search
hyperparams = np.logspace(-4, 4, 50)
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values
kfolds = 5

# Write your code here ...
import pandas as pd

#from q2_2 import cross_validation_linear_regression

# Define a range of alpha values for hyperparameter search
hyperparams = np.logspace(-4, 4, 50)
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values
kfolds = 5

# Write your code here ...

best_hyperparam, best_mean_squared_error, mean_squared_errors = cross_validation_linear_regression(kfolds, hyperparams, X_train, y_train)

import matplotlib.pyplot as plt
#plt.plot(hyperparams)
plt.plot(hyperparams,mean_squared_errors)
plt.legend(['hyperparameters', 'mean_squared_errors'])
plt.xlabel('hyperparameters')
plt.ylabel('mean_squared_errors')
plt.title('hyperparameters vs mean_squared_errors')
plt.show()

