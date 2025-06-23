from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model (replace with your actual model file)
model = pickle.load(open('Student Mark Prediction.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    race = request.form['race']
    education = request.form['education']
    lunch = request.form['lunch']
    prep = request.form['prep']
    reading_score = float(request.form['reading_score'])

    # Example encoding - you should use the same encoding used during model training
    input_data = [gender, race, education, lunch, prep, reading_score]

    # Dummy preprocessing - replace with your actual preprocessing pipeline
    # For now, let's say you use one-hot encoding + reading_score
    # Here we simulate this with dummy fixed-length numerical vector
    # Replace this with actual feature engineering
    X = preprocess_input(input_data)  # Define this function below or inline

    prediction = model.predict([X])[0]
    return render_template('result.html', prediction=round(prediction, 2))

# Dummy function to simulate input preprocessing
def preprocess_input(inputs):
    # Simple example: convert all to dummy values
    mapping = {
        'female': 0, 'male': 1,
        'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4,
        "bachelor's degree": 1, "some college": 4, "master's degree": 3,
        "associate's degree": 0, "high school": 2, "some high school": 5,
        "standard": 1, "free/reduced": 0,
        "none": 0, "completed": 1
    }
    encoded = [
        mapping.get(inputs[0], 0),
        mapping.get(inputs[1], 0),
        mapping.get(inputs[2], 0),
        mapping.get(inputs[3], 0),
        mapping.get(inputs[4], 0),
        inputs[5]  # reading score (float)
    ]
    return encoded

if __name__ == '__main__':
    app.run(debug=True)
