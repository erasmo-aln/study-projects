import numpy as np


def relu(input_data):
    output_data = np.maximum(0, input_data)

    return output_data


def relu_prime(input_data):
    input_data[input_data > 0] = 1
    input_data[input_data <= 0] = 0

    return input_data
