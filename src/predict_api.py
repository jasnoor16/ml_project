from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd
import logging
import threading
import time
import psutil

# Prometheus
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

# Logging Setup
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'api.log'))
    ]
)
logger = logging.getLogger("ml_app.api")

# Flask App
app = Flask(__name__)
metrics = PrometheusMetrics(app, path="/metrics")


# Custom Prometheus metrics
model_version = "1.0"
prediction_requests = Counter(
    'model_prediction_requests_total', 'Total prediction requests', ['model_version', 'status']
)

# Custom Prometheus error metric
prediction_error_requests = Counter(
    'model_prediction_requests_error_total', 'Total prediction error requests', ['model_version']
)

prediction_time = Histogram(
    'model_prediction_duration_seconds', 'Time taken to predict', ['model_version']
)
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage in percent')

# Load models
model_v1_path = os.path.join("models", "Linear_Regression.pkl")
model_v2_path = os.path.join("models", "Random_Forest.pkl")
model_v1 = joblib.load(model_v1_path) if os.path.exists(model_v1_path) else None
model_v2 = joblib.load(model_v2_path) if os.path.exists(model_v2_path) else None
logger.info("Models loaded.")

# Load preprocessing artifacts
processed_dir = "./data/processed/"
column_transformer = joblib.load(os.path.join(processed_dir, "column_transformer.pkl"))
scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
encoder = joblib.load(os.path.join(processed_dir, "time_spent_label_encoder.pkl"))

# Convert time
def convert_time_spent(value):
    if value <= 30:
        return "0 - 30 Minutes"
    elif 30 < value <= 60:
        return "30 - 60 Minutes"
    elif 60 < value <= 90:
        return "1 Hour - 1.5 Hours"
    else:
        return "2+ Hours"

# Prediction logic
def process_prediction(model):
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        if "features" not in data:
            prediction_error_requests.labels(model_version=model_version).inc()  # Increment error counter
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]
        expected_columns = [
            "Stake", "Ward/Branch", "# of Adult Volunteers", "# of Youth Volunteers", 
            "Distance", "Time Spent", "Completed More Than One Route", "Routes Completed", "Doors in Route"
        ]
        if len(features) != len(expected_columns):
            prediction_error_requests.labels(model_version=model_version).inc()  # Increment error counter
            return jsonify({"error": f"Expected {len(expected_columns)} features, got {len(features)}"}), 400

        feature_df = pd.DataFrame([features], columns=expected_columns)
        feature_df["Completed More Than One Route"] = feature_df["Completed More Than One Route"].map({"Yes": 1, "No": 0}).fillna(0)
        feature_df["Routes Completed"] = feature_df["Routes Completed"].map({"1": 1, "2": 2, "3": 3, "More than 3": 4}).fillna(1)
        feature_df["Time Spent"] = encoder.transform([convert_time_spent(feature_df["Time Spent"][0])])
        feature_encoded = column_transformer.transform(feature_df)
        feature_scaled = scaler.transform(feature_encoded)

        prediction = model.predict(feature_scaled)[0]

        # PROMETHEUS METRICS
        prediction_requests.labels(model_version=model_version, status="success").inc()
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)

        return jsonify({"predicted_donation_bags": int(prediction)})

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        prediction_error_requests.labels(model_version=model_version).inc()  # Increment error counter
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": str(e)}), 500

# Routes
@app.route('/')
def home():
    return "<h1>Welcome to the ML Model Prediction API</h1>"

@app.route('/ml_project_home', methods=['GET'])
def project_info():
    return jsonify({
        "message": "Welcome to the ML Model Prediction API!",
        "description": "This API predicts donation bags collected.",
        "available_endpoints": {
            "/v1/predict": "Model v1 (Linear Regression)",
            "/v2/predict": "Model v2 (Random Forest)",
            "/health_status": "Check API health",
            "/ml_project_home": "API Info"
        },
        "example_request_payload": {
            "features": ["Riverbend Stake", "Londonderry", 3, 2, 1.5, 30, "No", "2", 50]
        }
    })

@app.route('/health_status', methods=['GET'])
def health_status():
    return jsonify({"status": "API is running!"}), 200

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    if not model_v1:
        return jsonify({"error": "Model v1 not available"}), 500
    return process_prediction(model_v1)

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    if not model_v2:
        return jsonify({"error": "Model v2 not available"}), 500
    return process_prediction(model_v2)

# Background resource monitoring
def monitor_resources():
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)
        cpu_usage.set(process.cpu_percent())
        time.sleep(15)

from prometheus_client import REGISTRY, generate_latest

@app.route('/metrics')
def custom_metrics():
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    logger.info("Starting ML Prediction API on port 9999...")
    thread = threading.Thread(target=monitor_resources, daemon=True)
    thread.start()
    app.run(host='0.0.0.0', port=9999, debug=True)
