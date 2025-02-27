import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def save_model(model, model_name, save_dir="./models/"):
    """ Save the trained model to a file """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model and return MAE, RMSE, R2 Score """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2
