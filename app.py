import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load model and scaler
model = joblib.load("diabetes_model.pkl")  
scaler = joblib.load("scaler.pkl")        

app = Flask(__name__)
CORS(app)  # allow React Native app to access


# Test route for root
@app.route("/")
def home():
    return "SugarSense API is running!"  # This is what shows in the browser


# Feature order must match training
FEATURE_ORDER = [
    "GenHlth", "HighBP", "BMI", "Age", "HighChol", "CholCheck", "Income", "Sex",
    "HeartDiseaseorAttack", "HvyAlcoholConsump", "AnyHealthcare", "DiffWalk",
    "PhysActivity", "Smoker", "Veggies", "Fruits", "Education", "Stroke"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Build feature array in correct order
        features = np.array([[float(data[f]) for f in FEATURE_ORDER]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        return jsonify({"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
