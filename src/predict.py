# Import necessary libraries
import os
import numpy as np
import joblib
import yaml
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import pandas as pd

# Define directories
processed_dir = "./data/processed/"
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)  # Ensure results directory exists

# Load prediction configuration
with open("./configs/predict_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load test data
X_test = np.load(os.path.join(processed_dir, "X_test.npy"))  # Features
y_test = np.load(os.path.join(processed_dir, "y_test.npy"))  # Actual values

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8000")  # Ensure correct tracking server

# Function to get the latest MLflow run_id if not provided
def get_latest_run_id():
    client = mlflow.tracking.MlflowClient()

    # List all experiments
    experiments = client.search_experiments()
    print("Available Experiments:")
    for exp in experiments:
        print(f" - Experiment {exp.experiment_id}: {exp.name}")

    # Search for runs in the correct experiment
    runs = client.search_runs(experiment_ids=["757100552648002188"], order_by=["start_time desc"])

    if runs:
        latest_run_id = runs[0].info.run_id
        print(f"Using latest MLflow run_id: {latest_run_id}")
        return latest_run_id
    else:
        print("No valid MLflow runs found. Please manually enter a run_id.")
        return None  # Return None instead of raising an error

# Ask user for run_id
run_id = input("Enter the MLflow run_id (press Enter to use the latest run): ").strip()
if not run_id:
    run_id = get_latest_run_id()

if not run_id:
    print("No valid run_id provided. Exiting.")
    exit(1)

# Load model from MLflow
print(f"Loading model from MLflow run_id: {run_id}")
model_uri = f"runs:/{run_id}/Linear_Regression"  # Ensure correct model name
try:
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Make predictions
y_pred = model.predict(X_test)

# Save predictions as a .npy file
np.save(os.path.join(results_dir, "predictions.npy"), y_pred)

# Convert results into a DataFrame for easy viewing
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

# Show first 10 results
print("\nPredictions vs Actual Values:")
print(results_df.head(10))  # Display the first 10 rows

# Save results as a CSV file
results_df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

# Generate scatter plot (Actual vs. Predicted)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfect Prediction")
plt.xlabel("Actual Donation Bags Collected")
plt.ylabel("Predicted Donation Bags Collected")
plt.title("Actual vs. Predicted Donations (MLflow Model)")
plt.legend()
plt.savefig(os.path.join(results_dir, "scatter_plot_predictions.png"))
plt.show()

# Generate bar chart (First 10 Predictions)
plt.figure(figsize=(10, 5))
bar_width = 0.4
indices = np.arange(10)

plt.bar(indices, results_df["Actual"][:10], width=bar_width, label="Actual", color="blue")
plt.bar(indices + bar_width, results_df["Predicted"][:10], width=bar_width, label="Predicted", color="orange")

plt.xlabel("Test Sample Index")
plt.ylabel("Donation Bags Collected")
plt.title("Actual vs. Predicted Donations (First 10 Predictions)")
plt.xticks(indices + bar_width / 2, indices)  # Set x-axis labels
plt.legend()
plt.savefig(os.path.join(results_dir, "bar_chart_predictions.png"))
plt.show()

print("Predictions completed and saved in the results folder.")
