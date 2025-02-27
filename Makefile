# Define variables
VENV_DIR = .venv
PYTHON = python3

# Create virtual environment
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Run preprocessing
preprocess:
	$(VENV_DIR)/bin/python src/preprocess.py

# Train the model
train:
	$(VENV_DIR)/bin/python src/train.py

# Run predictions
predict:
	$(VENV_DIR)/bin/python src/predict.py

# Run everything
all: setup preprocess train predict
