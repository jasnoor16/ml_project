import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utils.model_utils import evaluate_model

class Evaluator:
    """ Evaluator class to handle model performance tracking """

    def __init__(self):
        """ Initialize paths and load test data """
        self.processed_dir = "./data/processed/"
        self.models_dir = "./models/"
        self.results_dir = "./results/"
        os.makedirs(self.results_dir, exist_ok=True)  # Ensure results directory exists

        # Load test data
        self.X_test = np.load(os.path.join(self.processed_dir, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.processed_dir, "y_test.npy"))

    def evaluate_and_plot(self, model_name):
        """ Evaluate a model and save its performance metrics """
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

    def run_evaluation(self):
        """ Run model evaluation for all models """
        print("\nEvaluating Models...")
        for model in ["linear_model", "random_forest_model", "decision_tree_model"]:
            self.evaluate_and_plot(model)

        print("✅ Model evaluation completed!")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
