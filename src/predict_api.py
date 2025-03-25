from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd
import logging

# Set log directory
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'api.log'))
    ]
)
logger = logging.getLogger("ml_app.api")

# Create Flask app
app = Flask(__name__)

# Define model paths
model_v1_path = os.path.join("models", "Linear_Regression.pkl")
model_v2_path = os.path.join("models", "Random_Forest.pkl")

# Load Model v1
if os.path.exists(model_v1_path):
    model_v1 = joblib.load(model_v1_path)
    logger.info("Model v1 (Linear Regression) loaded successfully.")
else:
    model_v1 = None
    logger.warning("Model v1 not found.")

# Load Model v2
if os.path.exists(model_v2_path):
    model_v2 = joblib.load(model_v2_path)
    logger.info("Model v2 (Random Forest) loaded successfully.")
else:
    model_v2 = None
    logger.warning("Model v2 not found.")

def convert_time_spent(value):
    if value <= 30:
        return "0 - 30 Minutes"
    elif 30 < value <= 60:
        return "30 - 60 Minutes"
    elif 60 < value <= 90:
        return "1 Hour - 1.5 Hours"
    else:
        return "2+ Hours"

# Function to process prediction 
def process_prediction(model):
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")

        if "features" not in data:
            logger.warning("Missing 'features' key in request.")
            return jsonify({"error": "Missing 'features' key in request JSON"}), 400

        features = data["features"]

        expected_columns = [
            "Stake", "Ward/Branch", "# of Adult Volunteers", "# of Youth Volunteers", 
            "Distance", "Time Spent", "Completed More Than One Route", "Routes Completed",
            "Doors in Route"
        ]
        if len(features) != len(expected_columns):
            logger.warning(f"Feature length mismatch: expected {len(expected_columns)}, got {len(features)}")
            return jsonify({"error": f"Expected {len(expected_columns)} features, but got {len(features)}"}), 400

        # Create DataFrame
        feature_df = pd.DataFrame([features], columns=expected_columns)

        # Load preprocessing artifacts
        processed_dir = "./data/processed/"
        column_transformer = joblib.load(os.path.join(processed_dir, "column_transformer.pkl"))
        scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
        encoder = joblib.load(os.path.join(processed_dir, "time_spent_label_encoder.pkl"))

        # Preprocessing: match what was done in training
        feature_df["Completed More Than One Route"] = feature_df["Completed More Than One Route"].map({"Yes": 1, "No": 0}).fillna(0)

        route_mapping = {"1": 1, "2": 2, "3": 3, "More than 3": 4}
        feature_df["Routes Completed"] = feature_df["Routes Completed"].map(route_mapping).fillna(1)

        feature_df["Time Spent"] = feature_df["Time Spent"].apply(convert_time_spent)
        feature_df["Time Spent"] = encoder.transform(feature_df["Time Spent"])


        # Apply one-hot encoding and scaling
        feature_encoded = column_transformer.transform(feature_df)
        feature_scaled = scaler.transform(feature_encoded)

        logger.info(f"Transformed input: {feature_scaled.tolist()}")

        prediction = model.predict(feature_scaled)
        logger.info(f"Prediction result: {prediction[0]}")

        return jsonify({"predicted_donation_bags": int(prediction[0])})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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
        logger.error("Model v1 is not loaded.")
        return jsonify({"error": "Model v1 (Linear Regression) is not available"}), 500
    return process_prediction(model_v1)

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    if not model_v2:
        logger.error("Model v2 is not loaded.")
        return jsonify({"error": "Model v2 (Random Forest) is not available"}), 500
    return process_prediction(model_v2)

if __name__ == '__main__':
    logger.info("Starting ML Prediction API on port 9999...")
    app.run(host='0.0.0.1', port=9999, debug=True)
