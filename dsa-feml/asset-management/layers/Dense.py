import numpy as np


class Dense:

    def __init__(self, feat_size, out_size):
        self.feat_size = feat_size
        self.out_size = out_size

        self.weights = (np.random.normal(0, 1, feat_size * out_size) * np.sqrt(2 / feat_size)).reshape(feat_size, out_size)
        self.bias = np.random.rand(1, out_size) - 0.5

        self.input_data = None
        self.output = None

    def forward(self, input_data):
        self.input_data = input_data
        self.output = np.dot(self.input_data, self.weights) + self.bias

        return self.output

    def backward(self, output_prime, learning_rate):
        input_prime = np.dot(output_prime, self.weights.T)
        weight_prime = np.dot(self.input_data.T.reshape(-1, 1), output_prime)

        self.weights = self.weights - (weight_prime * learning_rate)
        self.bias = self.bias - (output_prime * learning_rate)

        return input_prime
