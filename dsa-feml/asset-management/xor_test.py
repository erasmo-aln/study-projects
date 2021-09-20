import numpy as np

from layers.activation import ActivationLayer
from layers.dense import Dense
from layers.network import Network

from utils.activation_functions import relu, relu_prime
from utils.loss_functions import mse, mse_prime


X_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

X_train = X_train.reshape(-1, 2)
y_train = y_train.reshape(-1, 1)

xor_model = Network(mse, mse_prime)
xor_model.add(Dense(2, 3))
xor_model.add(ActivationLayer(relu, relu_prime))
xor_model.add(Dense(3, 1))

xor_model.fit(X_train, y_train, epochs=2000, learning_rate=0.01)

y_pred = xor_model.predict(X_train)

print(f'Output: {list(y_train.reshape(-1,))}')
print('-----')
print(f'Predicted: {[round(float(pred)) for pred in y_pred]}')
