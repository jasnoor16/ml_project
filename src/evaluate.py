import yaml
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load parameters from YAML
with open("configs/parameters.yml", "r") as f:
    params = yaml.safe_load(f)

# Set MLflow Tracking URI
mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

# Load test data
X_test = np.load("./data/processed/X_test.npy")
y_test = np.load("./data/processed/y_test.npy")

# Load model from MLflow using dynamic experiment name and run ID
experiment_name = params["mlflow"]["experiment_name"]
run_id = params["mlflow"].get("run_id", "your-default-run-id")  # Replace if needed
model_uri = f"runs:/{run_id}/Linear_Regression"

# Load the trained model
try:
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

print(f"Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
