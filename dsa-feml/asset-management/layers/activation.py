from utils.activation_functions import relu, relu_prime


class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

        self.input_data = None
        self.output = None

    def forward(self, input_data):
        self.input_data = input_data
        self.output = self.activation(self.input_data)

        return self.output

    def backward(self, output_prime, learning_rate):
        return (self.activation_prime(self.input_data) * output_prime)
