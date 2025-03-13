import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Define model URI
model_uri = f"runs:/{run_id}/Linear_Regression"  # Ensure correct model name

# Load test data
X_test = np.load("./data/processed/X_test.npy")
y_test = np.load("./data/processed/y_test.npy")

# Load the trained model
try:
    print(f"Loading model from MLflow run_id: {run_id}")
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
