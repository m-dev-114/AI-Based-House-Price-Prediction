from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load trained model and scaler
model = tf.keras.models.load_model("house_price_model.h5")
scaler = joblib.load("scaler.pkl")

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
        return jsonify({'predicted_price': round(float(prediction[0][0]), 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
