Docker Documentation for Alberta Food Drive ML Project

This project uses Docker to containerize both the Machine Learning prediction API and the MLflow tracking server. This allows the application to be deployed and run consistently on any system, regardless of local setup or operating system.

Why We Used Docker:

- To make sure the same environment runs across all machines
- To separate the ML API from MLflow tracking
- To avoid installing everything manually
- To support reproducibility, which is important in MLOps

What We Did:

1. We created two separate Dockerfiles:
   - Dockerfile.mlapp: For the Flask API that handles model predictions
   - Dockerfile.mlflow: For the MLflow server to track experiments and models

2. We created a docker-compose.yml file to run both containers together. This file connects both services using a shared network and assigns specific ports to each one.

3. We exposed ports:
   - 9999 for the Flask app
   - 8000 for the MLflow UI

4. We mounted volumes:
   - logs/ is mounted into the container to collect logs from the API and training script
   - data/processed/ is mounted to allow the API to access preprocessing artifacts (like encoders and scalers)

5. We set up environment variables:
   - LOG_DIR is used inside both Python scripts to store logs in the right place
   - MLFLOW_TRACKING_URI is passed so training logs go to the MLflow server running in the container

6. We used docker-compose up --build to build and launch everything together.

7. We tested all endpoints using curl and Postman to confirm that predictions and MLflow logging work correctly inside the container.

How to Build and Run:

Step 1: Run the following command in the project root:

docker-compose up --build

This builds the Docker images and starts both containers.

Once running:
- Open http://127.0.0.1:9999 to access the prediction API
- Open http://127.0.0.1:8000 to access MLflow UI

What to Do if Build Breaks:

Sometimes Docker build breaks due to cache or version issues. If that happens:

docker-compose down
docker system prune -a
docker-compose up --build

How We Built and Pushed Images to Docker Hub:

After testing locally, we built the Docker images manually and pushed them to Docker Hub for sharing.

Step 1: Log in

docker login

Step 2: Tag and push the Flask API image

docker build -f Dockerfile.mlapp -t jasnoor/ml-app:latest .
docker tag jasnoor/ml-app:latest docker.io/jasnoor/ml-app:latest
docker push docker.io/jasnoor/ml-app:latest

Step 3: Tag and push the MLflow image

docker build -f Dockerfile.mlflow -t jasnoor/mlflow:latest .
docker tag jasnoor/mlflow:latest docker.io/jasnoor/mlflow:latest
docker push docker.io/jasnoor/mlflow:latest

Docker Image Links (add after pushing):

Flask API Image: <paste-link-here>
MLflow Image: <paste-link-here>

Key Takeaways:

- Docker helped us isolate our application and run it smoothly without worrying about setup
- Using Docker Compose made it easier to manage both services (API and MLflow) at the same time
- This setup supports reproducibility, logging, and easy redeployment for future use

This Docker documentation summarizes what we implemented, why we did it, and how to run everything properly.
