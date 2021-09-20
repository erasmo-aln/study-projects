import numpy as np


def mse(y_true, y_pred):
    error = np.mean((y_pred - y_true)**2)

    return error


def mse_prime(y_true, y_pred):
    error_prime = (2 * (y_pred - y_true)) / y_true.size

    return error_prime
