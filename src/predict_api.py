from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Define model paths
model_v1_path = os.path.join("models", "Linear_Regression_pipeline.pkl")  # v1: Linear Regression Pipeline
model_v2_path = os.path.join("models", "Random_Forest_pipeline.pkl")  # v2: Random Forest Pipeline

# Load Model v1
if os.path.exists(model_v1_path):
    model_v1 = joblib.load(model_v1_path)
else:
    model_v1 = None

# Load Model v2
if os.path.exists(model_v2_path):
    model_v2 = joblib.load(model_v2_path)
else:
    model_v2 = None

# Function to process prediction 
def process_prediction(model):
    try:
        data = request.get_json()

        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request JSON"}), 400

        features = data["features"]

        # Expected feature format
        expected_columns = [
            "Stake", "Ward/Branch", "# of Adult Volunteers", "# of Youth Volunteers", 
            "Distance", "Time Spent", "Completed More Than One Route", "Routes Completed",
            "Doors in Route"
        ]
        
        if len(features) != len(expected_columns):
            return jsonify({"error": f"Expected {len(expected_columns)} features, but got {len(features)}"}), 400
        
        # Convert input into DataFrame
        feature_df = pd.DataFrame([features], columns=expected_columns)

        # Since the model pipeline includes preprocessing, we can directly make predictions
        prediction = model.predict(feature_df)

        return jsonify({"predicted_donation_bags": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Routes defined AFTER the process_prediction function
@app.route('/')
def home():
    return "<h1>Welcome to the ML Model Prediction API</h1>"

@app.route('/ml_project_home', methods=['GET'])
def project_info():
    return jsonify({
        "message": "Welcome to the ML Model Prediction API!",
        "description": "This API predicts the number of donation bags collected based on input features.",
        "available_endpoints": {
            "/v1/predict": "Predict donation bags collected using Model Version 1 (Linear Regression)",
            "/v2/predict": "Predict donation bags collected using Model Version 2 (Random Forest)",
            "/health_status": "Check API health status",
            "/ml_project_home": "API documentation and usage"
        },
        "example_request_payload": {
            "features": ["Stake", "Ward/Branch", 3, 2, 1, 30, "No", "2", 50]  
        }
    })

@app.route('/health_status', methods=['GET'])
def health_status():
    return jsonify({"status": "API is running!"}), 200

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    if not model_v1:
        return jsonify({"error": "Model v1 (Linear Regression) is not available"}), 500

    return process_prediction(model_v1)

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    if not model_v2:
        return jsonify({"error": "Model v2 (Random Forest) is not available"}), 500

    return process_prediction(model_v2)

# Run Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)
