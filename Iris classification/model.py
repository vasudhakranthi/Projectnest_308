import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pickle

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using pickle
with open('iris_classifier.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

print("Model saved as 'iris_classifier.pkl'.")

# Load the saved model (for demonstration)
with open('iris_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Test the loaded model with sample input
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input
sample_prediction = loaded_model.predict(sample_data)
print(f"Predicted Class for sample input: {iris.target_names[sample_prediction][0]}")
