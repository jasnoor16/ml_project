# ML Model Prediction API Documentation

This API predicts how many donation bags will be collected based on inputs like volunteers, distance, time, and routes. It is built using Flask and uses trained machine learning models.

## 1. How to Set Up and Run the API

### Requirements

- Python 3
- Flask, scikit-learn, joblib, pandas, numpy
- MLflow installed and running

### Step-by-Step Setup

Step 1: Clone the Repository

git clone https://github.com/jasnoor16/ml_project.git
cd ml_project

Step 2: Create Virtual Environment

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

Step 3: Run the Flask API

python src/predict_api.py

Access the API locally at:
http://127.0.0.1:9999

## 2. API Endpoints

/  
Method: GET  
Description: Basic welcome message  
curl -X GET http://127.0.0.1:9999/

/ml_project_home  
Method: GET  
Description: API usage info and available endpoints  
curl -X GET http://127.0.0.1:9999/ml_project_home

/health_status  
Method: GET  
Description: Health check for the API  
curl -X GET http://127.0.0.1:9999/health_status

## 3. Make a Prediction

You can predict donation bags using two versions of the model.

Version 1: Linear Regression  
URL: /v1/predict  
Method: POST  

curl -X POST http://127.0.0.1:9999/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1, 30, "No", "2", 50]}'

Sample Response:  
{"predicted_donation_bags": 42}

Version 2: Random Forest  
URL: /v2/predict  
Method: POST  

curl -X POST http://127.0.0.1:9999/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1, 30, "No", "2", 50]}'

Sample Response:  
{"predicted_donation_bags": 39}

## 4. Input Format

The "features" list must contain exactly 9 items in this order:

1. Stake (e.g., "Riverbend Stake")
2. Ward/Branch (e.g., "Clareview Ward")
3. # of Adult Volunteers (e.g., 1, 2, 3)
4. # of Youth Volunteers (e.g., 1, 2, 3)
5. Distance (numeric)
6. Time Spent (in minutes, numeric)
7. Completed More Than One Route ("Yes" or "No")
8. Routes Completed ("1", "2", "3", "More than 3")
9. Doors in Route (numeric)

The backend will handle preprocessing such as:
- Time category conversion
- Label encoding
- One-hot encoding
- Scaling

## 5. Error Handling

Missing Features  

curl -X POST http://127.0.0.1:9999/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1, 30]}'

Expected Response:  
{"error": "Expected 9 features, but got 6"}

Invalid Data Type  

curl -X POST http://127.0.0.1:9999/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": ["Riverbend Stake", "Clareview Ward", "three", 2, 1, 30, "No", "2", 50]}'

Expected Response:  
{"error": "could not convert string to float: 'three'"}

End of documentation.
