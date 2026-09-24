# Olist Late Delivery Prediction — MLOps

An end-to-end MLOps project for predicting whether an Olist Brazilian e-commerce order will be delivered late or on time.

## Project Overview

The project moves the machine learning workflow from notebooks to a production-oriented structure including data processing, validation, model registry, testing, API serving, containerization, CI/CD, and monitoring.

## Architecture

```text
PostgreSQL
    ↓
data_access.py
    ↓
ml_table
    ↓
Great Expectations
    ↓
Feature Engineering
    ↓
Saved Preprocessor
    ↓
MLflow Model Registry
    ↓
Weighted XGBoost Ensemble
    ↓
FastAPI
    ↓
Docker
    ↓
CI/CD + Monitoring
```

## Repository Structure

```text
.
├── app/                    # FastAPI application
├── artifacts/              # DVC-managed artifacts
├── config/                 # Configuration files
├── data/                   # Data-related files
├── logs/                   # Runtime logs
├── models/                 # Model configuration/artifacts used by the service
├── notebooks/              # Original development notebooks
├── src/                    # Reusable Python modules
├── tests/                  # Automated tests
├── .github/workflows/      # GitHub Actions CI workflow
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development/test dependencies
├── Dockerfile
├── docker-compose.yml
└── artifacts.dvc
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

The current local development setup uses a local DVC remote. A shared cloud/object-storage remote can be configured when collaborative data sharing is required.

## MLflow

MLflow is used for experiment/model management and Model Registry.

The FastAPI service loads the registered models using MLflow Model Registry URIs:

```text
models:/late_delivery_ensemble_best_f1/1
models:/late_delivery_ensemble_best_pr_auc/1
models:/late_delivery_ensemble_weight_5_25/1
```

The service therefore does not load the production models directly from the training notebooks.

## Running MLflow Locally

```bash
mlflow server --host 127.0.0.1 --port 5000 --allowed-hosts "localhost:*,127.0.0.1:*"
```

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

Returns model type, threshold, and feature count.

### Prediction

```http
POST /predict
```

Returns:

```json
{
  "prediction": "late",
  "probability_late": 0.62,
  "model_version": "1"
}
```

### Monitoring

```http
GET /metrics
```

Returns:

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

The current test suite covers data-access logic, feature engineering, model initialization, and data validation.

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

Run the complete local setup:

```bash
docker compose up --build
```

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

The latest completed CI run passed all stages.

## Monitoring

The API tracks:

* prediction request count
* prediction errors
* error rate
* average prediction latency

Prediction requests are also logged with prediction output, probability, latency, and model version.

## Reproducibility Notes

The preprocessing pipeline is loaded from a saved artifact rather than refitted during inference.

The final prediction uses the saved ensemble configuration, candidate feature list, model weights, and threshold.

The saved preprocessing artifact was created with scikit-learn 1.9.0, while the current runtime uses 1.9.1; this currently produces a warning and should be aligned in a future reproducibility cleanup.

## Project Goal

The goal of Task 3 is to transform the original notebook-based machine learning workflow into a reproducible, testable, deployable, and monitorable ML service.
