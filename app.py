import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load Model & Scaler
MODEL_PATH = "house_price_model.h5"
SCALER_PATH = "scaler.pkl"

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    model = None
    print("🚨 ERROR: Model file not found. Please retrain the model.")

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None
    print("🚨 ERROR: Scaler file not found. Please retrain the model.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler missing. Please train the model first.'})

    try:
        data = [float(x) for x in request.form.values()]
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data)
        return jsonify({'predicted_price': prediction[0][0]})
    except Exception as e:
        return jsonify({'error': str(e)})
