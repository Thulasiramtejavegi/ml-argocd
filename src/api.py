#!/usr/bin/env python3

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

# Load the model and scaler on startup
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/isolation_forest_model.joblib')
SCALER_PATH = os.environ.get('SCALER_PATH', '/app/models/scaler.joblib')

try:
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route("/", methods=["GET"])
def home():
    return """
    <h2>Welcome to the Anomaly Detection API</h2>
    <p>Use the <code>/predict</code> endpoint with a POST request to send features.<br>
    Try <a href="/health">/health</a> to check if the model is loaded.<br>
    Or test with a form at <a href="/predict-form">/predict-form</a>.</p>
    """

@app.route('/health', methods=['GET'])
def health():
    if model is not None and scaler is not None:
        return jsonify({"status": "healthy"}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "Model or scaler not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid request format. Expected 'features' field with data points."}), 400
        
        features = np.array(data['features'])
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        scores = model.decision_function(features_scaled)

        results = []
        for i in range(len(predictions)):
            results.append({
                "id": i,
                "prediction": "normal" if predictions[i] == 1 else "anomaly",
                "anomaly_score": float(-scores[i])
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-form", methods=["GET"])
def predict_form():
    return """
    <h2>Test Predict Endpoint</h2>
    <form action="/predict-form" method="post">
        <label for="features">Enter comma-separated features:</label><br>
        <input type="text" name="features" size="60"><br><br>
        <input type="submit" value="Submit">
    </form>
    """

@app.route("/predict-form", methods=["POST"])
def predict_form_post():
    try:
        raw = request.form.get("features", "")
        features = [float(x.strip()) for x in raw.split(",")]
        features_np = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_np)
        predictions = model.predict(features_scaled)
        scores = model.decision_function(features_scaled)

        prediction = "normal" if predictions[0] == 1 else "anomaly"
        anomaly_score = float(-scores[0])

        return f"""
        <h2>Prediction Result</h2>
        <p><strong>Prediction:</strong> {prediction}</p>
        <p><strong>Anomaly Score:</strong> {anomaly_score:.4f}</p>
        <a href="/predict-form">Try Again</a>
        """
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p><a href='/predict-form'>Try Again</a>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port)
