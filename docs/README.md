Alberta Food Drive ML Project

This machine learning project predicts how many donation bags will be collected during Alberta’s Food Drive campaigns. It uses real-world data such as number of volunteers, distance, time, and location. The model helps organizers plan better donation drives.

The project follows MLOps best practices: it is modular, reproducible, and ready for deployment. It uses MLflow for tracking experiments, DVC for data versioning, and Docker for containerization.

Team Members:
- Jasnoor Kaur Khangura
- Deeksha LNU
- Ravneet Singh Plaha
- Rahul Singla

Project Folder Structure:

ml_project/
├── data/                 # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/               # Saved trained model files
├── notebooks/            # EDA and prototyping notebooks
├── src/
│   ├── train.py
│   ├── predict_api.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── ml_utils/
│       ├── model_utils.py
│       └── helpers.py
├── configs/
│   ├── train_config.yaml
│   └── predict_config.yaml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── Dockerfile.mlapp
├── Dockerfile.mlflow
├── docker-compose.yml
└── .gitignore, .dvcignore

How to Run This Project Locally:

Step 1: Create a Virtual Environment

python -m venv .venv
source .venv/bin/activate         # For Mac/Linux
.venv\Scripts\activate            # For Windows

Step 2: Install Dependencies

pip install -r requirements.txt
pip install -r requirements-dev.txt  

DVC Setup:

Step 1: Initialize and Track Data

dvc init
dvc add data/raw
dvc add data/processed
git add .
git commit -m "Initialized DVC and tracked data"

Step 2: Setup Google Drive as Remote and Push

dvc remote add -d gdrive_remote gdrive://your_drive_id_here
dvc push

Training and Evaluation:

Step 1: Preprocess the Data

python src/preprocess.py

Step 2: Train the Models

python src/train.py

Step 3: Evaluate the Models

python src/evaluate.py

Step 4: (Optional) Run Makefile Commands

make all           # Run everything
make preprocess    # Only preprocessing
make train         # Only training
make evaluate      # Only evaluation
make predict       # Make predictions from CLI

MLflow Setup for Tracking:

Step 1: Start the MLflow UI

mlflow ui --port 8000

Step 2: Run Training Script (auto logs to MLflow)

python src/train.py

Step 3: Get Run ID and Serve Model 

mlflow models serve -m "runs:/<RUN_ID>/model"

Configuration Files:

configs/train_config.yaml
Contains model hyperparameters for training.

configs/predict_config.yaml
Contains default model and output directory setup.

Model Performance:

| Model              | MAE   | RMSE  | R² Score |
|--------------------|-------|-------|----------|
| Linear Regression  | 14.51 | 32.65 | 0.0696   |
| Random Forest      | 15.11 | 32.96 | 0.0515   |
| Decision Tree      | 21.95 | 38.14 | -0.2694  |

Final Model:
Linear Regression was chosen as the final model based on best performance.

Docker Usage:

This project uses Docker to run the MLflow server and Flask API in isolated containers.

To build and run the services:

docker-compose up --build

The Flask API will run on:
http://127.0.0.1:9999

The MLflow UI will run on:
http://127.0.0.1:8000

Docker Image Links :

ML App Docker Image: <paste-link-here>
MLflow Docker Image: <paste-link-here>

Git and DVC Version Control:

To push code changes:
git add .
git commit -m "Your message"
git push origin main

To push data changes:
dvc push

Google Drive stores the latest version of raw and processed data.

Project completed with full MLOps integration: preprocessing, training, evaluation, logging, versioning, and deployment — all automated and containerized.
