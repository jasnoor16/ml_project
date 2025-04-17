Edmonton Food Drive ML Project

This machine learning project predicts how many donation bags will be collected during Edmonton’s Food Drive campaigns. It uses real-world data such as number of volunteers, distance, time, and location. The model helps organizers plan better donation drives.

The project follows MLOps best practices: it is modular, reproducible, and ready for deployment. It uses MLflow for tracking experiments, DVC for data versioning, and Docker for containerization.

Team Members:
- Jasnoor Kaur Khangura
- Deeksha LNU
- Ravneet Singh Plaha
- Rahul Singla

## Technologies Used

- **Flask**: Lightweight API framework for handling prediction requests.
- **Prometheus**: Metrics collection and monitoring system to track the performance of the system.
- **Grafana**: Dashboard for visualizing system performance metrics in real-time.
- **MLflow**: Platform for managing the machine learning lifecycle, including experiment tracking and model deployment.
- **DVC**: Data versioning tool to keep track of large datasets and ensure reproducibility of experiments.
- **Docker**: Containerization tool to deploy the application in isolated environments, making it easier to set up and run the system across various environments.


---

Project Folder Structure:

.
├── Dockerfile.mlapp
├── Dockerfile.mlflow
├── Makefile
├── data
│   ├── external
│   │   └── External_property_data.csv
│   ├── processed
│   │   ├── X_test.npy
│   │   ├── X_train.npy
│   │   ├── cleaned_data_2023.csv
│   │   ├── cleaned_data_2024.csv
│   │   ├── column_transformer.pkl
│   │   ├── scaler.pkl
│   │   ├── time_spent_label_encoder.pkl
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   ├── processed.dvc
│   ├── raw
│   │   ├── data_2023.csv
│   │   └── data_2024.csv
│   └── raw.dvc
├── docker-compose.yml
├── docs
│   ├── API_documentation.md
│   ├── Docker_Guide.md
│   ├── README.md
│   ├── USAGE.md
│   └── monitoring.md
├── gdrive-creds.json
├── grafana
│   └── provisioning
│       ├── dashboards
│       │   ├── dashboard.yml
│       │   ├── predict_api_dashboard.json
│       │   └── training_dashboard.json
│       └── datasources
│           └── prometheus_datasource.yml
├── logs
│   ├── api.log
│   └── train.log
├── mlruns
│   └── models
├── models
│   ├── Decision_Tree.pkl
│   ├── Linear_Regression.pkl
│   └── Random_Forest.pkl
├── notebooks
│   ├── EDA_food_drive_2023.ipynb
│   ├── EDA_food_drive_2024.ipynb
│   ├── Merged_property_data.ipynb
│   ├── Model_Deployment_Streamlit.ipynb
│   └── Modelling_with 2023_on 2024.ipynb
├── predictions
│   └── predictions.npy
├── preprocessing
│   ├── column_transformer.pkl
│   ├── scaler.pkl
│   └── time_spent_label_encoder.pkl
├── prometheus
│   ├── prometheus.yml
│   └── rules
│       └── ml_alerts.yml
├── requirements.txt
├── results
│   ├── bar_chart_predictions.png
│   ├── decision_tree_model_performance.png
│   ├── linear_model_performance.png
│   ├── predictions.csv
│   ├── predictions.npy
│   ├── random_forest_model_performance.png
│   └── scatter_plot_predictions.png
├── run_with_metrics.sh
└── src
    ├── evaluate.py
    ├── ml_utils
    │   ├── __init__.py
    │   ├── helpers.py
    │   ├── model_utils.py
    │   └── monitoring.py
    ├── predict.py
    ├── predict_api.py
    ├── preprocess.py
    ├── start_monitoring_server.py
    ├── train.py
    └── trigger_predictions.py

22 directories, 64 files

How to Run This Project Locally:

Step 1: Create a Virtual Environment

python -m venv .venv
source .venv/bin/activate         # For Mac/Linux
.venv\Scripts\activate            # For Windows

Step 2: Install Dependencies

pip install -r requirements.txt

DVC Setup:

Step 1: Initialize and Track Data

dvc init
dvc add data/raw
dvc add data/processed
git add .
git commit -m "Initialized DVC and tracked data"

Step 2: Setup Google Drive as Remote and Push
Google Drive Credential Setup (For DVC):
To enable DVC to sync data with Google Drive, we used a service account credential file called gdrive-creds.json.

- This file authorizes secure access to the shared Google Drive folder used as a remote.

- It contains sensitive private keys, so we do not push it to GitHub.

- The file is included in .gitignore to protect our secrets and follow best security practices.

- If someone wants to run this project with DVC, they need to create their own gdrive-creds.json from a Google Cloud service account and place it in the root directory.

dvc remote modify gdrive_remote gdrive_use_service_account true
dvc remote modify gdrive_remote gdrive_service_account_json_file_path gdrive-creds.json
dvc remote add -d gdrive_remote gdrive://your_drive_id_here
dvc push

## Training and Evaluation:

**Step 1: Preprocess the Data**
This step cleans and prepares the data before training the model.
python src/preprocess.py

**Step 2: Train the Models**
Trains Linear Regression and Random Forest models.
python src/train.py

**Step 3: Evaluate the Models**
Tests each model on unseen data to evaluate performance (MAE, RMSE, R² score).
python src/evaluate.py

**Step 4: (Optional) Run Makefile Commands**

make all           # Run everything
make preprocess    # Only preprocessing
make train         # Only training
make evaluate      # Only evaluation
make predict       # Make predictions from CLI



MLflow Setup for Tracking:

Step 1: Start the MLflow UI

mlflow ui --port 8000

Step 2: Run Training Script (auto logs to MLflow)

python3 src/train.py

Step 3: Get Run ID and Serve Model 

mlflow models serve -m "runs:/<RUN_ID>/model"


Model Performance:

| Model              | MAE   | RMSE  | R² Score |
|--------------------|-------|-------|----------|
| Linear Regression  | 14.51 | 32.65 | 0.0696   |
| Random Forest      | 15.11 | 32.96 | 0.0515   |
| Decision Tree      | 21.95 | 38.14 | -0.2694  |

Final Model:
Linear Regression was chosen as the final model based on best performance.

Docker Usage:

To run the project in a containerized environment using Docker, refer to the [Docker Guide](Docker_Guide.md).

To build and run the services:

docker-compose up --build

The Flask API will run on:
http://127.0.0.1:9999

The MLflow UI will run on:
http://127.0.0.1:8000

Docker Image Links :

ML App Docker Image: <https://hub.docker.com/repository/docker/jasnoor709/docker-mlapp/general>
MLflow Docker Image: <https://hub.docker.com/repository/docker/jasnoor709/docker-mlflow/general>
or docker pull jasnoor709/mlflow:latest

Pull via CLI:
docker pull jasnoor709/docker-mlapp:latest  
docker pull jasnoor709/docker-mlflow:latest

## API Documentation

For detailed information on how to interact with the API, check out the [API Documentation](API_documentation.md).

## Monitoring and Metrics

The system integrates **Prometheus** to collect metrics and **Grafana** to visualize them in real-time dashboards. You can view live data such as CPU usage, memory usage, and prediction latency on the Grafana dashboards.

For more details, check the [Monitoring Guide](monitoring.md).

## Usage

Once the system is running, you can use the `/v1/predict` or `/v2/predict` endpoints to make predictions:

- **Endpoint**: `/v1/predict` (Linear Regression Model)
- **Request Method**: POST
- **Request Payload**:
    ```json
    {
      "features": ["Riverbend Stake", "Londonderry", 3, 2, 1.5, 30, "No", "2", 50]
    }
    ```

For more examples, refer to the [Usage Guide](Usage.md).

Git and DVC Version Control:

To push code changes:
git add .
git commit -m "Your message"
git push origin main

To push data changes:
dvc push

Google Drive stores the latest version of raw and processed data.

Project completed with full MLOps integration: preprocessing, training, evaluation, logging, versioning, and deployment — all automated and containerized.
