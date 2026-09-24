# Olist Late Delivery Prediction — MLOps

An end-to-end MLOps project for predicting whether an Olist Brazilian e-commerce order will be delivered late or on time.

## Project Overview

The project transforms a notebook-based machine learning workflow into a production-oriented structure covering data processing, validation, model management, testing, API serving, containerization, CI/CD, and monitoring.

## Architecture

### Training / Data Pipeline

```text
PostgreSQL
    ↓
Data Access
    ↓
ML Table
    ↓
Great Expectations
    ↓
Feature Engineering
    ↓
Saved Preprocessor
    ↓
Model Training / MLflow
```

### Inference / Serving Pipeline

```text
API Request
    ↓
FastAPI
    ↓
Feature Processing
    ↓
MLflow Model Registry
    ↓
Weighted XGBoost Ensemble
    ↓
Prediction + Probability
```

## Repository Structure

```text
.
├── app/                    # FastAPI application
├── artifacts/              # DVC-managed artifacts
├── config/                 # Configuration files
├── data/                   # Data-related files
├── logs/                   # Runtime logs
├── models/                 # Model configuration used by the service
├── notebooks/              # Original development notebooks
├── src/                    # Reusable Python modules
├── tests/                  # Automated tests
├── .github/workflows/      # GitHub Actions CI workflow
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development and testing dependencies
├── Dockerfile              # API image
├── Dockerfile.mlflow       # MLflow image
├── docker-compose.yml      # API + MLflow services
└── artifacts.dvc           # DVC metadata
```

## Environment Setup

Python 3.11 is used for the project.

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

Create a `.env` file for local environment variables such as database credentials.

## DVC

DVC is used to version large project artifacts.

The main tracked artifact directory is:

```text
artifacts/
```

The corresponding Git-tracked metadata file is:

```text
artifacts.dvc
```

Useful commands:

```bash
dvc status
dvc pull
dvc push
```

The current development environment uses a local DVC remote. This is suitable for the current local workflow, but a shared cloud/object-storage remote would be required for multi-machine collaboration.

## MLflow

MLflow is used for experiment and model management and for the Model Registry.

The FastAPI service loads the registered ensemble models using MLflow Model Registry URIs:

```text
models:/late_delivery_ensemble_best_f1/1
models:/late_delivery_ensemble_best_pr_auc/1
models:/late_delivery_ensemble_weight_5_25/1
```

The production inference code therefore loads models through MLflow Model Registry rather than directly from the training notebooks.

## Running MLflow Locally

For local development:

```bash
mlflow server --host 127.0.0.1 --port 5000 --allowed-hosts "localhost:*,127.0.0.1:*"
```

The Docker Compose setup uses a dedicated MLflow service with SQLite backend storage and local artifact storage.

## Running the API Locally

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health

```http
GET /health
```

Returns:

```json
{"status": "ok"}
```

### Model Information

```http
GET /model-info
```

Returns model configuration information such as model type, threshold, and feature count.

### Prediction

```http
POST /predict
```

Returns:

```json
{
  "prediction": "on_time",
  "probability_late": 0.17497,
  "model_version": "1"
}
```

### Monitoring

```http
GET /metrics
```

Returns runtime metrics such as:

```json
{
  "total_requests": 2,
  "total_errors": 0,
  "error_rate": 0.0,
  "average_latency_seconds": 0.6205
}
```

## Testing

Run all tests:

```bash
python -m pytest tests/ -v
```

The current test suite covers:

* data-access logic
* feature engineering
* model initialization
* data validation

The current local test suite passes all 7 tests.

## Code Quality

Run:

```bash
pre-commit run --all-files
```

Or individually:

```bash
black --check app src tests
flake8 --max-line-length=120 src/ app/ tests/
```

## Docker

Build the API image:

```bash
docker build -t late-delivery-api .
```

## Docker Compose

Run the API and MLflow services together:

```bash
docker compose up --build
```

The Compose setup provides:

```text
MLflow → port 5000
API    → port 8000
```

MLflow uses:

```text
mlflow.db
mlartifacts/
```

as local development storage.

These local MLflow state files are intentionally not stored in Git. Therefore, the current Compose configuration is intended for the configured local development environment; a fully portable multi-machine deployment would require a shared MLflow backend and artifact store.

## CI/CD

GitHub Actions runs automatically after a push.

The workflow checks:

```text
Black
↓
Flake8
↓
pytest
↓
Docker build
```

The latest completed CI workflow passed successfully.

## Monitoring

The API tracks:

* prediction request count
* prediction errors
* error rate
* average prediction latency

Prediction requests are also logged with prediction output, probability, latency, and model version.

## Reproducibility Notes

The preprocessing pipeline is loaded from a saved artifact rather than refitted during inference.

The final prediction uses the saved:

* ensemble configuration
* candidate feature list
* model weights
* prediction threshold

The saved preprocessing artifact was created with scikit-learn 1.9.0, while the current runtime uses 1.9.1. This currently produces an `InconsistentVersionWarning`; aligning the versions would be a future reproducibility cleanup.

The current DVC remote and MLflow backend/artifact storage are local-development resources rather than shared remote infrastructure.

## Project Goal

The goal of Task 3 is to transform the original notebook-based machine learning workflow into a modular, testable, deployable, and monitorable ML service.
