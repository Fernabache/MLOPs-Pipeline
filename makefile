.PHONY: install test lint clean deploy

# Variables
PYTHON := python3
PIP := pip3
TEST_PATH := tests
SRC_PATH := src

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	pytest $(TEST_PATH) --cov=$(SRC_PATH) --cov-report=term-missing

lint:
	flake8 $(SRC_PATH) tests
	black $(SRC_PATH) tests --check
	isort $(SRC_PATH) tests --check-only

format:
	black $(SRC_PATH) tests
	isort $(SRC_PATH) tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +

deploy-infra:
	cd infrastructure/terraform && \
	terraform init && \
	terraform apply -auto-approve

deploy-model:
	python scripts/deploy.py

setup-monitoring:
	python scripts/setup_monitoring.py

validate-data:
	python scripts/validate_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

all: clean install lint test
