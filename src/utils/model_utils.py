"""
This script contains utility functions for handling machine learning models.

It provides:
- Saving trained models to disk.
- Loading saved models for reuse.
- Evaluating model performance on test data.

These functions help streamline model management and evaluation.
"""

import os
import joblib  
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Define the directory where trained models will be stored
models_dir = "./models/"
os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists before saving models

def save_model(model, model_name):
    """
    Saves the trained machine learning model to a file in the models directory.
    
    This function allows easy reuse of trained models without retraining.

    Parameters:
    - model (sklearn model): The trained model to be saved.
    - model_name (str): The name of the model, used for the filename.

    Returns:
    - None
    """
    joblib.dump(model, os.path.join(models_dir, f"{model_name}.pkl"))

def load_model(model_name):
    """
    Loads a previously saved model from the models directory.

    This function ensures that the model exists before attempting to load it.

    Parameters:
    - model_name (str): The name of the model file (without .pkl extension).

    Returns:
    - The loaded model (sklearn model).

    Raises:
    - FileNotFoundError: If the model file does not exist.
    """
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model '{model_name}.pkl' not found in {models_dir}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained model on test data.

    This function calculates three key metrics:
    - MAE (Mean Absolute Error): Measures the average absolute error.
    - RMSE (Root Mean Squared Error): Penalizes larger errors more than MAE.
    - R² Score: Measures how well the model explains variance in the data.

    Parameters:
    - model (sklearn model): The trained model to be evaluated.
    - X_test (numpy array): Test set features.
    - y_test (numpy array): Actual test set labels.

    Returns:
    - Tuple (mae, rmse, r2): The three evaluation metrics.

    Example usage:
    ```
    mae, rmse, r2 = evaluate_model(trained_model, X_test, y_test)
    print(f"MAE: {mae}, RMSE: {rmse}, R² Score: {r2}")
    ```
    """
    y_pred = model.predict(X_test)  # Generate predictions using the test set
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate RMSE
    r2 = r2_score(y_test, y_pred)  # Calculate R² score
    return mae, rmse, r2  # Return the three evaluation metrics
