from sklearn.metrics import accuracy_score

from utils import preprocess, models


# Read the dataset
X, y = preprocess.read_dataset()

# Split into train/test
X_train, X_test, y_train, y_test = preprocess.split_dataset(X, y)

# Vectorize the dataset
X_train_vectorized, X_test_vectorized = preprocess.vectorize(X_train, X_test)

# Iterate over models
results = {'Voting': [], 'Stacking': []}
model_list = {'Voting': models.voting_classifier(), 'Stacking': models.stacking_classifier()}

for model_name, model in model_list.items():

    print(f'Fitting Model: {model_name}')
    model = model.fit(X_train_vectorized, y_train)

    print(f'Saving Predictions and Accuracy...')
    predictions = model.predict(X_test_vectorized)

    results[model_name].append(accuracy_score(y_test, predictions))
    print('\n---\n')

# Print Results
print('Results\n')
for model_name, result in results.items():
    print('---')
    print(f'Model Name: {model_name}')
    print(f'Accuracy: {result[0]*100:.2f}%\n')
