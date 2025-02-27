# **Alberta Food Drive ML Project**

## **Project Overview**
This project focuses on **predicting donation bag collection for the Alberta Food Drive** using machine learning techniques. The goal is to analyze various factors such as volunteer efforts, locations, and past trends to create an optimized prediction model. The project follows **best practices for reproducibility, modularity, and maintainability**, ensuring it can be deployed in a real-world setting.

Our team members:
- **Jasnoor Kaur Khangura**
- **Deeksha LNU**
- **Ravneet Singh Plaha**
- **Rahul Singla**

---

## **Project Structure**
The project follows **MLOps** structure, ensuring that the code is **modular, reusable, and scalable**. Below is the directory structure:


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
│   └── utils/           # Shared helper functions
│       ├── model_utils.py
│       └── helpers.py
├── configs/
│   ├── train_config.yaml
│   └── predict_config.yaml
├── docs/                # Documentation folder
│   ├── README.md        # Project documentation
│   ├── user_guide.md    # Step-by-step guide for using this project
├── requirements.txt     # Dependencies required to run the project
├── Makefile             # Automates setup and training
└── .gitignore           # Exclude unnecessary files (not required for now)

Installation Guide
To set up this project, follow these steps:


1. Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

2. Install Dependencies
Run the following command to install all required libraries:

pip install -r requirements.txt

**Usage Guide**

1. Data Preprocessing
Run the preprocessing script to clean and format the data:

python src/preprocess.py
This script reads cleaned_data_2023.csv and cleaned_data_2024.csv.
It converts categorical data into numerical values.
It saves processed data in data/processed/.

2. Model Training
Train the machine learning models using:

python src/train.py
This script loads the processed training data.
It trains Linear Regression, Random Forest, and Decision Tree models using the hyperparameters in configs/train_config.yaml.
The trained model is saved in the models/ directory.

3. Model Evaluation
Evaluate the trained models using:

python src/evaluate.py
This script loads test data and evaluates the models.
It calculates MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² Score for each model.
The results are displayed in the terminal.

4. Making Predictions
Make predictions using the best-performing model:

python src/predict.py
The script loads the trained model from models/.
It predicts donation bag collections for 2024.
The predictions are saved in data/processed/predictions.npy.

Configuration Files
The project uses YAML configuration files for flexibility.

1. Training Configuration (configs/train_config.yaml)
This file contains the hyperparameters for each model. Example:

yaml

random_forest:
  n_estimators: 100
  max_depth: Null
  random_state: 42

decision_tree:
  max_depth: Null
  random_state: 42
This allows easy modification of model parameters without changing the code.

2. Prediction Configuration (configs/predict_config.yaml)
This file specifies which trained model to use for predictions:

yaml

default_model: "linear"
output_directory: "./data/processed/"
The default model can be changed to "linear" or "decision_tree".

Makefile Automation
To simplify execution, a Makefile is provided.

## Run Preprocessing, Training, and Evaluation Together
## RUN make all
make all

Runs preprocessing, model training, and evaluation.
Run Individual Steps

make preprocess  # Only runs preprocessing
make train       # Only trains the models
make evaluate    # Only evaluates models
make predict     # Only makes predictions

Model Performance
After evaluation, the model performances were:

Model	MAE	RMSE	R² Score
Linear Regression	14.51	32.65	0.0696
Random Forest	15.11	32.96	0.0515
Decision Tree	21.95	38.14	-0.2694
Based on these metrics, Linear Regression performed the best.

Final Model Selection
Since Linear Regression has the lowest RMSE and highest R² Score, it was chosen as the final model.
The trained model is saved as:
models/linear_model.pkl
