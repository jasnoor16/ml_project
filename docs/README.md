# **Alberta Food Drive ML Project**

## **Project Overview**
This project focuses on **predicting donation bag collection for the Alberta Food Drive** using machine learning techniques. The goal is to analyze various factors such as volunteer efforts, locations, and past trends to create an optimized prediction model. The project follows **best practices for reproducibility, modularity, and maintainability**, ensuring it can be deployed in a real-world setting.

### **Team Members**
- **Jasnoor Kaur Khangura**
- **Deeksha LNU**
- **Ravneet Singh Plaha**
- **Rahul Singla**

## **Project Structure**
The project follows **MLOps principles**, ensuring that the code is **modular, reusable, and scalable**. Below is the directory structure:

ml_project/
├── data/                # Stores datasets (raw, processed, external)
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/              # Saved trained model files (e.g., .pkl, .joblib)
├── notebooks/           # Jupyter notebooks for exploration/demos
├── src/
│   ├── train.py         # Script to train the model
│   ├── predict.py       # Script to make predictions
│   ├── preprocess.py    # Data preprocessing logic
│   ├── evaluate.py      # Model evaluation script
│   └── ml_utils/           # Shared helper functions
│       ├── model_utils.py
│       └── helpers.py
├── configs/
│   ├── train_config.yaml
│   └── predict_config.yaml
├── docs/                # Documentation folder
│   ├── README.md        # Project documentation
│   ├── user_guide.md    # Step-by-step guide for using this project
├── .dvcignore           # Files to ignore for DVC
├── requirements.txt     # Dependencies required to run the project
├── Makefile             # Automates setup and training
└── .gitignore           # Exclude unnecessary files

## **Installation Guide**
Follow these steps to set up and run the project:

### **1. Create a Virtual Environment**
python3 -m .venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

### **2. Install Dependencies**
pip install -r requirements.txt

### **3. Initialize DVC and Track Data**
dvc init
dvc add data/raw
dvc add data/processed
git add .
git commit -m "Initialized DVC and tracked data"
git push origin main

### **4. Configure DVC Remote Storage (Google Drive)**
dvc remote add -d gdrive_remote gdrive://1gNqMDp2dxr521HgqHt2B0iALIdOhTEIh
dvc push

## **Usage Guide**
### **1. Data Preprocessing**
python3 src/preprocess.py

### **2. Model Training**
python3 src/train.py

### **3. Model Evaluation**
python3 src/evaluate.py

### **4. Making Predictions**
python3 src/predict.py

## **MLflow Setup and Experiment Tracking**
### **1. Start MLflow UI**
mlflow ui --port 8000

### **2. Retrieve Run ID**
mlflow models serve -m "runs:/<RUN_ID>/model"
Replace `<RUN_ID>` with the actual **run ID** from the MLflow UI.

## **Configuration Files**
### **1. Training Configuration (`configs/train_config.yaml`)**
random_forest:
  n_estimators: 100
  max_depth: Null
  random_state: 42

decision_tree:
  max_depth: Null
  random_state: 42

### **2. Prediction Configuration (`configs/predict_config.yaml`)**
default_model: "linear"
output_directory: "./data/processed/"

## **Makefile Automation**
### **Run Everything (Preprocessing, Training, Evaluation)**
make all

### **Run Individual Steps**
make preprocess  # Runs preprocessing
make train       # Trains the models
make evaluate    # Evaluates models
make predict     # Runs predictions

## **Model Performance**
| Model              | MAE   | RMSE  | R² Score |
|--------------------|------|------|---------|
| **Linear Regression** | **14.51** | **32.65** | **0.0696** |
| **Random Forest** | 15.11 | 32.96 | 0.0515 |
| **Decision Tree** | 21.95 | 38.14 | -0.2694 |

## **Final Model Selection**
Since **Linear Regression** had the lowest RMSE and the highest R² Score, it was selected as the final model.

models/linear_model.pkl

## **Git Version Control**
### **1. Push Code Changes to GitHub**
git add .
git commit -m "Updated training and evaluation scripts"
git push origin main

### **2. Push Data Changes to DVC**
dvc push

## **DVC Remote Storage Link**
Our DVC-tracked dataset is stored in **Google Drive** and can be accessed using the configured remote storage.



### ** Project Completed Successfully!**
