import numpy as np


from q3_1 import compute_gradient_ridge,compute_gradient_simple
from q1_1 import rmse, linear_regression_predict


def gradient_descent_regression(X, y, reg_type='simple', hyperparameter=0.0, learning_rate=0.01, num_epochs=100):
    """
    Solves regression tasks using full-batch gradient descent.

    Parameters:
    X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
    y (np.ndarray): Target values of shape (n_samples,).
    reg_type (str): Type of regression ('simple' for simple linear, 'ridge' for ridge regression).
    hyperparameter (float): Regularization parameter, used only for ridge regression.
    learning_rate (float): Learning rate for gradient descent.
    num_epochs (int): Number of epochs for gradient descent.

    Returns:
    w (np.ndarray): Final weights after gradient descent optimization.
    b (float): Final bias after gradient descent optimization.
    """

    # Write your code here ...
    m = X.shape[1]
    w = np.zeros((m))
    b = 10.0

    if reg_type=='simple':
        for i in range(num_epochs):
            grad_w, grad_b = compute_gradient_simple(X,y,w,b)
            w = w - learning_rate*grad_w
            b = b - learning_rate*grad_b
            bias = np.array([b for _ in range(len(y))])
            y_hat = bias+linear_regression_predict(X,w)
            score = rmse(y_hat,y)
            print(f'[epoch: {i+1}] ========> score: {score:.2f}')
    elif reg_type=='ridge':
        for i in range(num_epochs):
            grad_w, grad_b = compute_gradient_ridge(X,y,w,b,hyperparameter)
            w = w -learning_rate*grad_w
            b = b - learning_rate*grad_b
            bias = [b for _ in range(len(y))]
            y_hat = bias+linear_regression_predict(X,w)
            score = rmse(y_hat,y)
            print(f'[epoch: {i+1}] ========> score: {score:.2f}')
    else:
        print(f'Uknown regression type {reg_type}')
        return None,None
    return w, b
