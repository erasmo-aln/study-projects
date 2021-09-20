import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from data.generate_data import generate_data

from layers.network import Network
from layers.dense import Dense
from layers.activation import ActivationLayer

from utils.loss_functions import mse, mse_prime
from utils.activation_functions import relu, relu_prime


# Create dataset if not exists
if not os.path.exists('data/asset_data.csv'):
    generate_data()

# Get the dataset
df = pd.read_csv('data/asset_data.csv', index_col=0)

# Split into training and test sets
train_data = df.iloc[:round(len(df) * 0.8)]
test_data = df.iloc[len(train_data):]

X_train = train_data.drop(columns='Sim_1_Call').values
y_train = train_data['Sim_1_Call'].values

X_test = test_data.drop(columns='Sim_1_Call').values
y_test = test_data['Sim_1_Call'].values

# Scale values
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = Network(mse, mse_prime)

model.add(Dense(5, 200))
model.add(ActivationLayer(relu, relu_prime))

model.add(Dense(200, 200))
model.add(ActivationLayer(relu, relu_prime))

model.add(Dense(200, 200))
model.add(ActivationLayer(relu, relu_prime))

model.add(Dense(200, 200))
model.add(ActivationLayer(relu, relu_prime))

model.add(Dense(200, 1))

# Training
model.fit(X_train, y_train, epochs=100, learning_rate=0.001)

# Get predictions
y_pred = model.predict(X_test)
y_pred = np.array([float(i) for i in y_pred]).reshape(-1,)

# Plot predicted and real values
comparison_data = pd.DataFrame({
    'Real': y_test,
    'Predicted': y_pred}, index=test_data.index)
mse_error = round(mean_squared_error(comparison_data.Real, comparison_data.Predicted), 3)

ax = sns.lineplot(data=comparison_data, palette='bright')

ax.set_xlabel('Date')
ax.set_ylabel('Asset Price')
ax.set_title(f'Epochs: 100 --- MSE: {mse_error}')

plt.show()
