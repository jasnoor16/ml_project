from flask import Flask, jsonify, request
import joblib
import os
import numpy as np
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load Preprocessing Artifacts
processed_dir = "./data/processed/"
column_transformer = joblib.load(os.path.join(processed_dir, "column_transformer.pkl"))
scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
encoder = joblib.load(os.path.join(processed_dir, "time_spent_label_encoder.pkl"))

# Define model path for Linear Regression (Model v1)
model_v1_path = os.path.join("models", "Linear_Regression.pkl")

# Load Model v1
if os.path.exists(model_v1_path):
    model_v1 = joblib.load(model_v1_path)
else:
    model_v1 = None

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
            "/v2/predict": "Predict donation bags collected using Model Version 2",
            "/health_status": "Check API health status",
            "/ml_project_home": "API documentation and usage"
        },
        "example_request_payload": {
            "features": ["Stake", "Ward/Branch", 3, 2, 1, 30, "No", "2", 50]  
            # No "Comments" column anymore
        }
    })

@app.route('/health_status', methods=['GET'])
def health_status():
    return jsonify({"status": "API is running!"}), 200

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    if not model_v1:
        return jsonify({"error": "Model v1 (Linear Regression) is not available"}), 500

    try:
        data = request.get_json()
        
        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request JSON"}), 400
        
        features = data["features"]
        
        # Expected feature format (excluding "Comments")
        expected_columns = [
            "Stake", "Ward/Branch", "# of Adult Volunteers", "# of Youth Volunteers", 
            "Distance", "Time Spent", "Completed More Than One Route", "Routes Completed",
            "Doors in Route"
        ]
        
        if len(features) != len(expected_columns):
            return jsonify({"error": f"Expected {len(expected_columns)} features, but got {len(features)}"}), 400
        
        # Convert input into DataFrame
        feature_df = pd.DataFrame([features], columns=expected_columns)

        # Drop "Comments" column if the model is still expecting it
        if "Comments" in column_transformer.feature_names_in_:
            feature_df["Comments"] = "No Comments"  # Assign a default value
            feature_df.drop(columns=["Comments"], inplace=True)

        # Convert categorical "Yes/No" values
        feature_df["Completed More Than One Route"] = feature_df["Completed More Than One Route"].map({"Yes": 1, "No": 0}).fillna(0)

        # Convert Routes Completed values
        route_mapping = {"1": 1, "2": 2, "3": 3, "More than 3": 4}
        feature_df["Routes Completed"] = feature_df["Routes Completed"].map(route_mapping).fillna(1)

        # Apply column transformer
        feature_encoded = column_transformer.transform(feature_df)

        # Scale numerical variables
        feature_scaled = scaler.transform(feature_encoded)

        # Make prediction
        prediction = model_v1.predict(feature_scaled)

        return jsonify({"predicted_donation_bags": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)
