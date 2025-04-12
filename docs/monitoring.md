# Monitoring & Observability Report – Lab 5

## Monitoring Strategy

We integrated Prometheus and Grafana into our Docker-based machine learning project. The monitoring setup helps track system health, prediction API performance, and training process metrics. We used the `prometheus_flask_exporter` package inside the prediction API and exposed metrics during model training using a custom monitoring module. All metrics are scraped by Prometheus and visualized using Grafana dashboards that are automatically provisioned at container startup.

## Key Metrics Tracked

### For the Prediction API
- `model_prediction_requests_total`: Tracks the total number of prediction requests categorized by status (success or error). This helps monitor the overall health and usage of the API.
- `model_prediction_duration_seconds`: Measures how long each prediction takes, useful for identifying latency issues.
- `app_memory_usage_bytes`: Shows how much memory the API container is using.
- `app_cpu_usage_percent`: Tracks how much CPU the API is consuming.

### For Training Scripts
- `regression_mean_squared_error`: Shows how close the predicted values are to the actual target.
- `regression_root_mean_squared_error`: Similar to MSE but in the same unit as the target variable.
- `regression_r_squared`: Indicates how well the model explains the variability of the data.
- `feature_importance`: Displays the most important features in the model.
- `tree_max_depth`: Helps us understand the complexity of tree-based models.
- `ensemble_tree_count`: Tracks the number of trees used in models like Random Forest.

## How Monitoring Helps

- Helps detect prediction failures early by watching error counts.
- Lets us track latency and spot performance issues over time.
- Shows CPU and memory usage to avoid crashes due to resource limits.
- During training, monitoring helps confirm whether the model is improving or not.
- Detects data drift or model degradation by comparing new metrics with previous ones.

## Alerts Configured

We configured alert rules in `ml_alerts.yml` to notify us when something goes wrong:

- **HighErrorRate**: Triggers when more than 10% of prediction requests fail in the last 5 minutes.
- **SlowPredictionResponse**: Triggers if the 95th percentile prediction duration exceeds 1 second.
- **HighMemoryUsage**: Triggers if memory usage exceeds 1.5 GB.
- **StalledTraining**: Triggers if no training metrics have been updated in the past 15 minutes.

These alerts help us take action before the system fails or performance drops significantly.
