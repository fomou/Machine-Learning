import numpy as np
from typing import List, Tuple

from q1_1 import rmse
from q2_1 import ridge_regression_optimize


def cross_validation_linear_regression(k_folds: int, hyperparameters: List[float],
                                       X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
    """
    Perform k-fold cross-validation to find the best hyperparameter for Ridge Regression.

    Args:
        k_folds (int): Number of folds to use.
        hyperparameters (List[float]): List of floats containing the hyperparameter values to search.
        X (np.ndarray): Numpy array of shape [observations, features].
        y (np.ndarray): Numpy array of shape [observations, 1].

    Returns:
        best_hyperparam (float): Value of the best hyperparameter found.
        best_mean_squared_error (float): Best mean squared error corresponding to the best hyperparameter.
        mean_squared_errors (List[float]): List of mean squared errors for each hyperparameter.
    """

    # Write your code here ...

    fold_size = X.shape[0]//k_folds

    folds = []
    mean_squared_errors = np.zeros(len(hyperparameters))
    best_mean_squared_error = np.inf
    for i in range(k_folds):
        validation = X[fold_size*i:(i+1)*fold_size]
        y_val = y[fold_size*i:(i+1)*fold_size]
        train = np.concatenate([X[:i*fold_size],X[(i+1)*fold_size:]])
        y_train = np.concatenate([y[:i*fold_size],y[(i+1)*fold_size:]])
        folds.append((train,y_train,validation,y_val))

    for i,t in enumerate(hyperparameters):
      rmses = []
      print(f'\nhyperparameters {t:.5f} : ')
      for k,(x,y,x_test,y_test) in enumerate(folds):
        w = ridge_regression_optimize(y,x,t)
        y_hat = np.matmul(x_test,w)
        e= rmse(y_test,y_hat)       
        rmses.append(e)
        if e < best_mean_squared_error:
          best_mean_squared_error = e
        print(f'\tfold {k+1} ===> RMSE: {e:.5f}')
      M_RMSE = np.mean(rmses)
      mean_squared_errors[i] = M_RMSE
      print(f'hyperparameters {t:.5f} ==> Mean of RMSE: {M_RMSE:.4f} ')
    best_ind = np.argmin(mean_squared_errors)
    best_hyperparam = hyperparameters[best_ind]

    return best_hyperparam, best_mean_squared_error, mean_squared_errors.tolist()
