import os
import yaml
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from utils.model_utils import save_model, evaluate_model

class Trainer:
    """ Trainer class to handle model training and evaluation """

    def __init__(self):
        """ Initialize paths and load data """
        self.processed_dir = "./data/processed/"
        self.models_dir = "./models/"
        self.results_dir = "./results/"
        os.makedirs(self.models_dir, exist_ok=True)  # Ensure models directory exists
        os.makedirs(self.results_dir, exist_ok=True)  # Ensure results directory exists

        # Load preprocessed training data
        self.X_train = np.load(os.path.join(self.processed_dir, "X_train.npy"))
        self.y_train = np.load(os.path.join(self.processed_dir, "y_train.npy"))

        # Load preprocessed test data
        self.X_test = np.load(os.path.join(self.processed_dir, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.processed_dir, "y_test.npy"))

        # Load YAML configuration file
        with open("./configs/train_config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

    def train_models(self):
        """ Train machine learning models and save them """

        print("Training Linear Regression Model...")
        linear_model = LinearRegression()
        linear_model.fit(self.X_train, self.y_train)
        save_model(linear_model, "linear_model")

        print("Training Random Forest Model...")
        rf_params = self.config["random_forest"]
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(self.X_train, self.y_train)
        save_model(rf_model, "random_forest_model")

        print("Training Decision Tree Model...")
        dt_params = self.config["decision_tree"]
        dt_model = DecisionTreeRegressor(**dt_params)
        dt_model.fit(self.X_train, self.y_train)
        save_model(dt_model, "decision_tree_model")

        print("✅ Model training completed!")

    def evaluate_and_plot(self, model_name):
        """ Evaluate models and generate performance plots """
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ Model {model_name} not found. Skipping...")
            return

        model = joblib.load(model_path)
        mae, rmse, r2 = evaluate_model(model, self.X_test, self.y_test)

        print(f"\n{model_name} Performance:")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R² Score: {r2:.4f}")

        # Generate and save bar plot
        plt.figure(figsize=(6, 4))
        metrics = {"MAE": mae, "RMSE": rmse, "R² Score": r2}
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'red', 'green'])
        plt.ylabel("Error")
        plt.title(f"{model_name} Performance")
        plt.savefig(os.path.join(self.results_dir, f"{model_name}_performance.png"))
        plt.close()

    def run_training_and_evaluation(self):
        """ Run the full training and evaluation pipeline """
        self.train_models()

        print("\nEvaluating Models...")
        for model in ["linear_model", "random_forest_model", "decision_tree_model"]:
            self.evaluate_and_plot(model)

        print("✅ Training and evaluation completed!")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_training_and_evaluation()
