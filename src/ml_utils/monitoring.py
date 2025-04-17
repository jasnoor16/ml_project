from prometheus_client import make_wsgi_app, Gauge
from wsgiref.simple_server import make_server
import threading
import logging

logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, port=8002):
        self.port = port
        self.started = False

    def start_server(self):
        if not self.started:
            print(f"✅ Starting Prometheus WSGI monitoring server on port {self.port}...")
            logger.info(f"✅ Starting Prometheus WSGI monitoring server on port {self.port}...")

            def launch():
                try:
                    app = make_wsgi_app()
                    httpd = make_server("0.0.0.0", self.port, app)
                    print(f"🌐 Prometheus WSGI server is now listening on port {self.port}")
                    httpd.serve_forever()
                except Exception as e:
                    print(f"❌ Failed to start Prometheus WSGI server: {e}")
                    logger.error(f"❌ Failed to start Prometheus WSGI server: {e}")

            threading.Thread(target=launch, daemon=True).start()
            self.started = True
        else:
            print("ℹ️ Prometheus server already running.")
            logger.info("ℹ️ Prometheus server already running.")

class RegressionMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)

        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')

        self.feature_importance = Gauge(
            'feature_importance', 'Feature importance value', ['feature_name']
        )

    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        print("📊 Recording regression metrics to Prometheus...")
        if mse is not None:
            self.mse.set(mse)
            print(f"🧮 MSE: {mse}")
        if rmse is not None:
            self.rmse.set(rmse)
            print(f"🧮 RMSE: {rmse}")
        if mae is not None:
            self.mae.set(mae)
            print(f"🧮 MAE: {mae}")
        if r_squared is not None:
            self.r_squared.set(r_squared)
            print(f"🧮 R² Score: {r_squared}")

        if feature_importance is not None:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for name, value in sorted_features:
                print(f"⭐ Feature: {name}, Importance: {value}")
                self.feature_importance.labels(feature_name=name).set(value)
