import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MLflow Tracking Setup (directly set here instead of parameters.yml)
mlflow_tracking_uri = "http://127.0.0.1:8000"
experiment_name = "ML_Project_Training"
run_id = "ee0a595275c948ac8d0800c9e3d1be7b"  # Change this to the actual run ID from MLflow UI

mlflow.set_tracking_uri(mlflow_tracking_uri)
model_uri = f"runs:/{run_id}/Linear_Regression"

# Load test data
X_test = np.load("./data/processed/X_test.npy")
y_test = np.load("./data/processed/y_test.npy")

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
