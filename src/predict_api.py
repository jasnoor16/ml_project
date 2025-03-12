from flask import Flask, jsonify

# Create Flask app
app = Flask(__name__)

# Home endpoint
@app.route('/')
def home():
    return "<h1>Welcome to the ML Model Prediction API</h1>"

# API Information Endpoint
@app.route('/ml_project_home', methods=['GET'])
def project_info():
    return jsonify({
        "message": "Welcome to the ML Model Prediction API!",
        "description": "This API serves machine learning models through REST endpoints. "
                       "It allows users to send data and receive predictions using machine learning models.",
        "available_endpoints": {
            "/v1/predict": "Predict using Model Version 1",
            "/v2/predict": "Predict using Model Version 2",
            "/health_status": "Check API health status",
            "/ml_project_home": "API documentation and usage"
        },
        "example_request_payload": {
            "features": [5.1, 3.5, 1.4, 0.2]  # Example feature values for a prediction request
        }
    })


@app.route('/health_status', methods=['GET'])
def health_status():
    return jsonify({"status": "API is running!"}), 200





# Run Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)
