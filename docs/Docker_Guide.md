## Docker Guide for Edmonton Food Drive ML Project

In this section, we’ll walk you through how we used **Docker** and **Docker Compose** to run all components of our machine learning project in isolated containers. This ensures everything works seamlessly and is easy to deploy and scale.

### What We Did

We containerized every major component of the project, including the **Flask API**, **MLflow server**, **Prometheus monitoring**, and **Grafana dashboards**. By using **Docker Compose**, we orchestrate all these containers with a single command.

### Docker Setup

1. **Install Docker**:

   First, make sure **Docker** and **Docker Compose** are installed on your machine. Follow the official guides if you haven't installed them yet: 
   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. **Dockerfile Setup**:

   - **Flask API**: This container runs our API which serves the predictions. We used the following Dockerfile for the **Flask API** (`Dockerfile.mlapp`):
   
     ```dockerfile
     FROM python:3.8-slim
     WORKDIR /app
     COPY requirements.txt /app/
     RUN pip install --no-cache-dir -r requirements.txt
     COPY . /app
     CMD ["python3", "src/predict_api.py"]
     ```

     This Dockerfile installs all the necessary Python dependencies and runs the `predict_api.py` script, which serves the predictions from our models.

   - **MLflow Server**: For model tracking and experiment management, we containerized **MLflow** in another Dockerfile (`Dockerfile.mlflow`):

     ```dockerfile
     FROM python:3.8-slim
     WORKDIR /app
     COPY requirements.txt /app/
     RUN pip install --no-cache-dir -r requirements.txt
     COPY . /app
     CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "8000"]
     ```

     This starts the **MLflow UI** on port `8000` to track and manage our model experiments.

3. **Prometheus and Grafana Setup**:

   - **Prometheus**: We used the official **Prometheus** image to collect metrics about system performance. Here’s the setup in **docker-compose.yml**:

     ```yaml
     prometheus:
       image: prom/prometheus:v2.52.0
       ports:
         - "9090:9090"
       volumes:
         - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
     ```

     This tells **Prometheus** to collect metrics about the system and expose them on port `9090`.

   - **Grafana**: We also used **Grafana** to visualize the data collected by **Prometheus**. The setup in `docker-compose.yml` is:

     ```yaml
     grafana:
       image: grafana/grafana:7.5.0
       ports:
         - "3000:3000"
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin
     ```

     This will start **Grafana** on port `3000`, and you can log in using `admin` as the username and password.

4. **Running All Containers**:

   The real magic happens when we bring all these components together using **Docker Compose**. The `docker-compose.yml` file manages all four services (`Flask API`, `MLflow`, `Prometheus`, and `Grafana`):

   ```yaml
   version: '3'
   services:
     app:
       build:
         context: .
         dockerfile: Dockerfile.mlapp
       ports:
         - "9999:9999"
       depends_on:
         - mlflow
         - prometheus
         - grafana

     mlflow:
       build:
         context: .
         dockerfile: Dockerfile.mlflow
       ports:
         - "8000:8000"

     prometheus:
       image: prom/prometheus:v2.52.0
       ports:
         - "9090:9090"
       volumes:
         - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

     grafana:
       image: grafana/grafana:7.5.0
       ports:
         - "3000:3000"
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin

   This docker-compose.yml file tells Docker to:

   Build the Flask API image from Dockerfile.mlapp

   Build the MLflow image from Dockerfile.mlflow

   Start Prometheus and Grafana from their official images

## Starting the Containers:

To start all the containers, run this simple command:
docker-compose up --build

This command will:

- Build the Docker images

- Start all the containers

- Map the necessary ports for each container:

Flask API: http://localhost:9999

MLflow UI: http://localhost:8000

Prometheus UI: http://localhost:9090

Grafana UI: http://localhost:3000

**Monitoring with Metrics:**
We also included the Prometheus metrics to monitor the performance of the Flask API. The /metrics endpoint, available at http://localhost:9999/metrics, serves the metrics data.

The Prometheus container is set up to collect data like memory usage, CPU usage, and prediction latency. Grafana fetches these metrics and visualizes them in real-time on dashboards.

**Running with Metrics:**
You can run the API with metrics collection by executing the run_with_metrics.sh script:

./run_with_metrics.sh
This script starts the Prometheus and Flask API servers with metrics collection enabled.

Stop the containers using:
docker-compose down


## Conclusion
With Docker and Docker Compose, we were able to successfully containerize and orchestrate all the services required for this ML project. Flask API, MLflow, Prometheus, and Grafana are all running in isolated containers, and we can monitor the performance of our system through Grafana in real-time.
