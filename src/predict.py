# Import necessary libraries
import os
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
import pandas as pd

# Define directories
processed_dir = "./data/processed/"
models_dir = "./models/"
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)  # Ensure results directory exists

# Load prediction configuration
with open("./configs/predict_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load test data
X_test = np.load(os.path.join(processed_dir, "X_test.npy"))  # Features
y_test = np.load(os.path.join(processed_dir, "y_test.npy"))  # Actual values

# Load the best model (Linear Regression)
model_name = "linear_model.pkl"
model_path = os.path.join(models_dir, model_name)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_name}' not found in {models_dir}")

model = joblib.load(model_path)

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
print("\n📌 Predictions vs Actual Values:")
print(results_df.head(10))  # Display the first 10 rows

# Save results as a CSV file
results_df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

# Generate scatter plot (Actual vs. Predicted)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfect Prediction")
plt.xlabel("Actual Donation Bags Collected")
plt.ylabel("Predicted Donation Bags Collected")
plt.title("Actual vs. Predicted Donations (Linear Regression)")
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

print("✅ Predictions completed and saved in the results folder!")
