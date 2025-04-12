#!/bin/bash

docker rm -f training-metrics 2>/dev/null

echo "Starting training with metrics monitoring..."

docker run \
  --name training-metrics \
  --network=ml_project_ml-network \
  ml_project-ml-app \
  python src/train.py

echo "TRAINING COMPLETED!!"
