class Network:

    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        result = []

        for _, example in enumerate(input_data):
            layer_output = example

            for layer in self.layers:
                layer_output = layer.forward(layer_output)

            result.append(layer_output)

        return result

    def fit(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(1, epochs + 1):
            error = 0

            for index, train_example in enumerate(X_train):
                layer_output = train_example

                for layer in self.layers:
                    layer_output = layer.forward(layer_output)

                error += self.loss(y_train[index], layer_output)

                gradient = self.loss_prime(y_train[index], layer_output)

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

            error = error / len(X_train)

            print(f'Epoch {epoch}/{epochs} --- MSE: {error}')
