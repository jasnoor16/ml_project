# Monitoring & Observability Report

## Monitoring Strategy

In this project, we integrated **Prometheus** and **Grafana** for real-time monitoring and observability. These tools help us track **system health**, **prediction API performance**, and **training process metrics**. We used the `prometheus_flask_exporter` package to expose metrics from the **Flask API**, and a custom **monitoring module** to expose training metrics. Prometheus scrapes the metrics, while **Grafana** visualizes them in real-time dashboards.

We also added additional functionality through the **`start_monitoring_server.py`** to start a monitoring server and the **`trigger_predictions.py`** to automate prediction requests for monitoring purposes.

## Key Metrics Tracked

### For the Prediction API

- **`model_prediction_requests_total`**:
  - **Description**: Tracks the total number of prediction requests, categorized by status (success or error).
  - **Why We Track It**: Helps monitor the health and usage of the prediction API. By tracking both success and failure, we can detect issues early.
  
- **`model_prediction_duration_seconds`**:
  - **Description**: Measures how long each prediction takes, capturing the time from when a request is made until a response is sent.
  - **Why We Track It**: Identifying latency issues is critical to maintaining an efficient API. Slow predictions can impact user experience and indicate performance bottlenecks.

- **`app_memory_usage_bytes`**:
  - **Description**: Tracks the memory usage of the prediction API container.
  - **Why We Track It**: Monitoring memory usage helps prevent crashes due to memory leaks or resource exhaustion. This ensures the API remains responsive.

- **`app_cpu_usage_percent`**:
  - **Description**: Tracks the percentage of CPU used by the prediction API.
  - **Why We Track It**: High CPU usage could indicate inefficient code or high traffic. By tracking it, we can prevent performance degradation or system crashes due to CPU spikes.

### For Training Scripts

- **`regression_mean_squared_error`**:
  - **Description**: Measures how close the predicted values are to the actual target values. A lower value indicates better predictions.
  - **Why We Track It**: MSE helps evaluate how well the model is performing in terms of error.

- **`regression_root_mean_squared_error`**:
  - **Description**: Similar to MSE, but provides the error in the same unit as the target variable.
  - **Why We Track It**: RMSE is more interpretable than MSE, and it gives us a sense of the model's performance in the context of the target variable.

- **`regression_r_squared`**:
  - **Description**: Indicates how well the model explains the variability of the data. A value closer to 1 means a better fit.
  - **Why We Track It**: R² helps us understand how much of the target’s variance is explained by the model, giving insight into model performance.

- **`feature_importance`**:
  - **Description**: Displays the most important features that the model uses to make predictions.
  - **Why We Track It**: Knowing which features are most influential helps improve model transparency and decision-making.

- **`tree_max_depth`**:
  - **Description**: Tracks the maximum depth of decision trees used in models like Random Forest.
  - **Why We Track It**: This metric helps us understand the complexity of the model, particularly for tree-based models. A deeper tree can lead to overfitting.

- **`ensemble_tree_count`**:
  - **Description**: Tracks the number of trees used in ensemble models like Random Forest.
  - **Why We Track It**: This metric gives insight into the size and complexity of the ensemble model, which can impact performance and overfitting.

## How Monitoring Helps

- **Early Detection of Failures**: Monitoring error counts helps us detect prediction failures early, allowing us to address issues before they affect users.
- **Performance Tracking**: By tracking latency, we can spot performance issues in real-time and optimize the API for faster predictions.
- **Resource Utilization**: Memory and CPU usage monitoring ensures that the system doesn’t crash due to resource limitations. It helps in scaling the system efficiently.
- **Model Training Monitoring**: During training, we track metrics like **MSE**, **RMSE**, and **R²** to confirm the model is improving over time. This ensures we’re on the right path and helps in debugging if training is stalling.
- **Detecting Data Drift/Model Degradation**: By comparing current metrics to previous values, we can identify data drift or model degradation, prompting retraining or adjustments to the model.

## Alerts Configured

We’ve configured several alert rules in **`ml_alerts.yml`** to notify us if something goes wrong:

- **HighErrorRate**: 
  - **Condition**: Triggers when the error rate exceeds 0% over the last 1 minute (i.e., when any errors occur in the last minute).
  - **Purpose**: Helps identify and address issues when the prediction API starts returning errors, ensuring timely intervention before system performance degrades.

- **SlowPredictionResponse**: 
  - **Condition**: Triggers if the 95th percentile prediction duration exceeds 1 second.
  - **Purpose**: Alerts us when the API response time is slow, helping us optimize the performance of our prediction service.

- **HighMemoryUsage**: 
  - **Condition**: Triggers if memory usage exceeds 1.5 GB.
  - **Purpose**: Ensures that the system doesn’t run out of memory, preventing potential crashes due to high memory consumption.

- **StalledTraining**: 
  - **Condition**: Triggers if no training metrics have been updated in the last 15 minutes.
  - **Purpose**: Helps us detect when the training process has stalled or failed, so we can restart or debug the training pipeline.

### Why We Chose These Metrics

We carefully selected these metrics because they provide a **comprehensive view** of the system’s performance and health. The combination of **API performance monitoring**, **system resource tracking**, and **model training evaluation** ensures that we can quickly detect and address issues. These metrics align with MLOps best practices for maintaining a **reliable and efficient machine learning system**.

## Files Added for Monitoring:

1. **`start_monitoring_server.py`**:
   - This script starts the **Prometheus monitoring server** in the background, allowing us to track **training metrics** (like MSE, R²) and **prediction performance** in real-time.

2. **`trigger_predictions.py`**:
   - This script is used to **trigger prediction requests** periodically, allowing us to simulate user traffic and monitor the **prediction API** performance.

## Conclusion

By integrating **Prometheus** for metrics collection and **Grafana** for visualization, we have set up a robust monitoring system for both the **prediction API** and the **training pipeline**. The **alerts** ensure we are notified when system performance degrades or when errors occur, allowing us to take action before it affects the users. This monitoring setup enhances the **reliability**, **scalability**, and **maintainability** of our system, adhering to MLOps best practices.


