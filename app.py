import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import joblib

# Disable GPU for Render deployment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Load trained model
model = keras.models.load_model("house_price_model.h5")

# Load pre-trained scaler
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from form
        data = [float(x) for x in request.form.values()]
        data = np.array(data).reshape(1, -1)

        # Scale input data
        data = scaler.transform(data)

        # Make prediction
        prediction = model.predict(data)

        # Return JSON response
        return jsonify({'predicted_price': float(prediction[0][0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Ensure correct port for Render
    app.run(host='0.0.0.0', port=port, debug=True)
