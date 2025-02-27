import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from ml_utils.model_utils import save_model, evaluate_model  # No 'src' needed

class Trainer:
    def __init__(self):
        """Initialize paths and load data"""
        self.processed_dir = "./data/processed/"
        self.models_dir = "./models/"
        os.makedirs(self.models_dir, exist_ok=True)

        # Load preprocessed data
        try:
            self.X_train = np.load(os.path.join(self.processed_dir, "X_train.npy"))
            self.y_train = np.load(os.path.join(self.processed_dir, "y_train.npy"))
            self.X_test = np.load(os.path.join(self.processed_dir, "X_test.npy"))
            self.y_test = np.load(os.path.join(self.processed_dir, "y_test.npy"))
            print("✅ Data loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            exit(1)

        # Load configuration
        try:
            with open("./configs/train_config.yaml", "r") as file:
                self.config = yaml.safe_load(file)
            print("✅ Configuration loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            exit(1)

        # Set up MLflow Tracking
        mlflow.set_tracking_uri("http://127.0.0.1:8000")
        mlflow.set_experiment("ML_Project_Training")

    def train_model(self, model_name, model, params=None):
        """ Train a model and log it to MLflow """
        print(f"🔹 Training {model_name}...")
        with mlflow.start_run(run_name=model_name):
            if params:
                mlflow.log_params(params)

            model.fit(self.X_train, self.y_train)
            save_model(model, model_name)

            mlflow.sklearn.log_model(model, model_name)

            mae, rmse, r2 = evaluate_model(model, self.X_test, self.y_test)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2_Score", r2)

    def train_models(self):
        """ Train all models """
        print("\n🚀 Starting Model Training...\n")

        linear_model = LinearRegression()
        self.train_model("Linear_Regression", linear_model)

        rf_params = self.config.get("random_forest", {})
        rf_model = RandomForestRegressor(**rf_params)
        self.train_model("Random_Forest", rf_model, rf_params)

        dt_params = self.config.get("decision_tree", {})
        dt_model = DecisionTreeRegressor(**dt_params)
        self.train_model("Decision_Tree", dt_model, dt_params)

        print("\n✅ All models trained successfully!\n")

    def run_training(self):
        """ Run training pipeline """
        self.train_models()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_training()
    print("\n🎯 Training & Evaluation Completed! Check MLflow UI.")
