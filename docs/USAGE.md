# Usage Guide - Edmonton Food Drive ML Project

This guide will walk you through how to set up, run, and interact with the **Edmonton Food Drive Machine Learning Project**. The project predicts the number of donation bags collected during food drives, using machine learning models like Linear Regression and Random Forest. We use **Flask** for the API, **Docker** for containerization, and **MLflow** for tracking experiments. **Prometheus** and **Grafana** are used for real-time monitoring of the system.

---

## 1. Set Up the Project

### **Requirements**
- **Python 3.x**
- Flask
- scikit-learn
- joblib
- pandas
- numpy
- MLflow (for model tracking)
- Docker (for containerization)
- DVC (for data versioning)

---

### **Step-by-Step Setup**

#### 1.1 Clone the Repository

```bash
git clone https://github.com/jasnoor16/ml_project.git
cd ml_project

#### 1.2 Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate  # For Mac/Linux
.venv\Scripts\activate     # For Windows
pip install -r requirements.txt

## 2. Run the Project Locally
#### 2.1 Start Flask API
python3 src/predict_api.py
The Flask API will be available at: http://127.0.0.1:9999

## 3. Running the Project in Docker Containers
We use Docker Compose to run the project in containers. The containers include the Flask API, MLflow, Prometheus, and Grafana.

#### 3.1 Build and Run the Docker Containers
docker-compose up --build

This command will:

Build all Docker images

Start the containers

Map the necessary ports for each container:

Flask API: http://localhost:9999

MLflow UI: http://localhost:8000

Prometheus UI: http://localhost:9090

Grafana UI: http://localhost:3000

## 4. Interact with the API
You can use Postman or curl to interact with the API and get predictions.

#### 4.1 Endpoints
Root Endpoint (GET):
Returns a welcome message.

curl -X GET http://127.0.0.1:9999/


API Home (GET):
Provides API usage information.
curl -X GET http://127.0.0.1:9999/ml_project_home

Health Status (GET):
Checks if the API is running.
curl -X GET http://127.0.0.1:9999/health_status


#### 4.2 Make a Prediction
To make predictions, you can use either Linear Regression or Random Forest models.

Linear Regression Model:
curl -X POST http://127.0.0.1:9999/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1.5, 30, "No", "2", 50]}'

Random Forest Model:
curl -X POST http://127.0.0.1:9999/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"features": ["Riverbend Stake", "Clareview Ward", 3, 2, 1.5, 30, "No", "2", 50]}'

## 5. Monitoring and Metrics
We use Prometheus to track system performance metrics, including:

Prediction Latency: Time taken for each prediction.

Memory Usage: The amount of memory the API container is using.

CPU Usage: The percentage of CPU used by the Flask API.

Prediction Errors: Total number of prediction errors over a given time.

Grafana visualizes these metrics in real-time dashboards.

**Conclusion**
The Prediction API allows you to predict the number of donation bags collected during the Edmonton Food Drive campaigns. By making a POST request to /v1/predict (Linear Regression) or /v2/predict (Random Forest), you can get real-time predictions based on the provided features. The API also includes error handling, so it returns informative messages if any required fields are missing or invalid.

You can monitor the API and model performance in real-time using Prometheus and Grafana dashboards.

