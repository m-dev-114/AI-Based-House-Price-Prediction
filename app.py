import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask App
app = Flask(__name__)

# Check if Model & Scaler Exist
MODEL_PATH = "house_price_model.h5"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    print("🚨 ERROR: Model file not found. Please retrain the model.")
    exit(1)

if not os.path.exists(SCALER_PATH):
    print("🚨 ERROR: Scaler file not found. Please retrain the model.")
    exit(1)

# Load Model and Scaler
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data)
        
        return jsonify({'predicted_price': round(prediction[0][0], 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

