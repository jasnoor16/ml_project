#!/bin/bash

# Remove existing training container if any
docker rm -f training-metrics 2>/dev/null 

echo "Starting training with metrics monitoring..."

# Run new training container with fixed name
docker run \
  --name training-metrics \
  --network=ml_project_ml-network \
  ml_project-ml-app \
  python src/train.py

echo "TRAINING COMPLETED!!"