# ML Model Prediction API Documentation

This API predicts the number of donation bags collected based on input features. It is built using Flask and serves machine learning models.

## 1. Setting Up and Running the API

### Prerequisites
Ensure you have the following installed:
- Python3
- Flask, NumPy, Pandas, Joblib, Scikit-learn
- MLflow for model tracking

### Step 1: Clone the Repository
git clone <https://github.com/jasnoor16/ml_project.git>
cd ml_project

### Step 2: Set Up Virtual Environment
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

### Step 3: Start the API
python src/predict_api.py
# The API will run at:
# http://127.0.0.1:9999

## 2. Available Endpoints

### Home Endpoint
# URL: /
# Method: GET
curl -X GET http://127.0.0.1:9999/

### API Information Endpoint
# URL: /ml_project_home
# Method: GET
curl -X GET http://127.0.0.1:9999/ml_project_home

### Health Check
# URL: /health_status
# Method: GET
curl -X GET http://127.0.0.1:9999/health_status

## 3. Making Predictions

### Predict Donation Bags (Version 1)
# URL: /v1/predict
# Method: POST
# Example Request:
curl -X POST http://127.0.0.1:9999/v1/predict -H "Content-Type: application/json" -d '{
    "features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1, 30, "No", "2", 50]
}'

# Example Response:
{
    "predicted_donation_bags": 42
}

### Predict Donation Bags (Version 2)
# URL: /v2/predict
# Method: POST
# Example Request:
curl -X POST http://127.0.0.1:9999/v2/predict -H "Content-Type: application/json" -d '{
    "features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1, 30, "No", "2", 50]
}'

# Example Response:
{
    "predicted_donation_bags": 39
}

## 4. Valid Input Values

# Stake and Ward names:
# - Stake: "Riverbend Stake", "Bonnie Doon Stake", "Gateway Stake", "YSA Stake", "Edmonton North Stake"
# - Ward/Branch: "Clareview Ward", "Woodbend Ward", "Londonderry Ward", "Clareview Ward"...

# Other feature values:
# - "# of Adult Volunteers": 1, 2, 3, ...
# - "# of Youth Volunteers": 1, 2, 3, ...
# - Distance: 1, 2, 3, ...
# - Time Spent: 15, 30, 45, ...
# - Completed More Than One Route: "Yes", "No"
# - Routes Completed: "1", "2", "3", "More than 3"
# - Doors in Route: 10, 20, 30, ...

## 5. Handling Errors

### Example: Missing Features
curl -X POST http://127.0.0.1:9999/v1/predict -H "Content-Type: application/json" -d '{
    "features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1, 30]
}'
# Response:
{
    "error": "Expected 9 features, but got 6"
}

### Example: Invalid Data Type
curl -X POST http://127.0.0.1:9999/v1/predict -H "Content-Type: application/json" -d '{
    "features": ["Riverbend Stake", "Clareview Ward", "three", 2, 1, 30, "No", "2", 50]
}'
# Response:
{
    "error": "could not convert string to float: 'three'"
}



# End of documentation.
