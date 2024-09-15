import numpy as np


# Part A:
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Returns the design matrix with an all one column appended

    Args:
        X (np.ndarray): Numpy array of shape [observations, num_features]

    Returns:
        np.ndarray: Numpy array of shape [observations, num_features + 1]
    """

    # Write your code here ...
    # ref: https://stackoverflow.com/questions/40814383/adding-a-column-in-front-of-a-numpy-array

    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias


# Part B:
def linear_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes $y = Xw$

    Args:
        X (np.ndarray): Numpy array of shape [observations, features]
        w (np.ndarray): Numpy array of shape [features, 1]

    Returns:
        np.ndarray: Numpy array of shape [observations, 1]
    """
    # Write your code here ...
    y = np.matmul(X,w)
    return y


# Part C:
def linear_regression_optimize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Optimizes MSE fit of $y = Xw$

    Args:
        y (np.ndarray): Numpy array of shape [observations, 1]
        X (np.ndarray): Numpy array of shape [observations, features]

    Returns:
        Numpy array of shape [features, 1]
    """

    # Write your answer here ....
    Xt = np.transpose(X)
    XtX = np.matmul(Xt,X)
    XtX_inv = np.linalg.inv(XtX)
    XtX_invXt = np.matmul(XtX_inv, Xt)
    w = np.matmul(XtX_invXt,y)
    return w


# Part D
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Evaluate the RMSE between actual and predicted values.

    Parameters:
    y (list or np.array): The actual values.
    y_hat (list or np.array): The predicted values.

    Returns:
    float: The RMSE value.
    """
    # Write your code here ...


    rmse_err = np.sqrt(np.sum((np.array(y)-np.array(y_hat))**2)/len(y))
    return rmse_err

