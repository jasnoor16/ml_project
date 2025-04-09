import sys
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ml_utils.model_utils import evaluate_model
import logging
from src.ml_utils.monitoring import RegressionMonitor  # ✅ NEW

# Set log directory
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'train.log' if __name__ == '__main__' and 'train' in __file__ else 'api.log'))
    ]
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self):
        """Initialize paths and load data"""
        self.processed_dir = "./data/processed/"
        self.models_dir = "./models/"
        os.makedirs(self.models_dir, exist_ok=True)

        # ✅ Initialize monitoring
        self.monitor = RegressionMonitor(port=8002)
        self.monitor.start_server()

        # Load preprocessed data
        try:
            self.X_train = np.load(os.path.join(self.processed_dir, "X_train.npy"))
            self.y_train = np.load(os.path.join(self.processed_dir, "y_train.npy"))
            self.X_test = np.load(os.path.join(self.processed_dir, "X_test.npy"))
            self.y_test = np.load(os.path.join(self.processed_dir, "y_test.npy"))
            logger.info("Data loaded successfully.")
            print("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            print(f"Error loading data: {e}")
            exit(1)

        # MLflow Tracking Setup
        self.mlflow_tracking_uri = "http://mlflow:8000"
        self.experiment_name = "ML_Project_Training"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Enable MLflow Auto-Logging
        mlflow.autolog()

        # Model Parameters
        self.random_forest_params = {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42
        }
        self.decision_tree_params = {
            "max_depth": None,
            "random_state": 42
        }

    def train_model(self, model_name, model, params=None):
        """Train a model and log it to MLflow"""
        logger.info(f"Training {model_name}...")
        print(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            if params:
                mlflow.log_params(params)

            # Create pipeline with imputer
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', model)
            ])

            pipeline.fit(self.X_train, self.y_train)

            # Save the model pipeline
            if model_name in ["Linear_Regression", "Random_Forest", "Decision_Tree"]:
                model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                joblib.dump(pipeline, model_path)

            mlflow.sklearn.log_model(pipeline, model_name)

            # Evaluate model
            mae, rmse, r2 = evaluate_model(pipeline, self.X_test, self.y_test)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2_Score", r2)

            # ✅ Record metrics with Prometheus monitor
            self.monitor.record_metrics(
                mse=rmse**2,     # MSE = RMSE squared
                rmse=rmse,
                mae=mae,
                r_squared=r2,
                feature_importance=None  # Add if you calculate it later
            )

        logger.info(f"{model_name} training completed.")
        print(f"{model_name} training completed.")

    def train_models(self):
        """Train all models"""
        logger.info("Starting Model Training...")
        print("Starting Model Training...")

        self.train_model("Linear_Regression", LinearRegression())
        self.train_model("Random_Forest", RandomForestRegressor(**self.random_forest_params))
        self.train_model("Decision_Tree", DecisionTreeRegressor(**self.decision_tree_params))

        logger.info("All models trained and saved.")
        print("All models trained and saved.")

    def run_training(self):
        """Run training pipeline"""
        self.train_models()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_training()
    logger.info("Training completed.")
    print("Training completed.")
