from flask import Flask, jsonify

# Create Flask app
app = Flask(__name__)

# Home endpoint
@app.route('/')
def home():
    return "<h1>Welcome to the ML Model Prediction API</h1>"

# API Information Endpoint
@app.route('/ml_project_home')
def project_info():
    return jsonify({
        "message": "This API serves machine learning models through REST endpoints.",
        "endpoints": {
            "/v1/predict": "Predict using Model Version 1",
            "/v2/predict": "Predict using Model Version 2",
            "/health_status": "Check API health status",
            "/ml_project_home": "API documentation and usage"
        },
        "note": "Send JSON payload to /v1/predict or /v2/predict for predictions."
    })

# Run Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)
