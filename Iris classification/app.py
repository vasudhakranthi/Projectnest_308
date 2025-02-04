from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
with open('iris_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form input
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare data for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = classes[prediction[0]]

        return render_template('index.html', prediction_text=f'Predicted Class: {predicted_class}')
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter valid numbers.')

if __name__ == "__main__":
    app.run(debug=True)
