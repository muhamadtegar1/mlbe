# BentoML Deployment Guide

## Overview
This project uses BentoML to serve Random Forest and CatBoost models for bank marketing prediction.

## Prerequisites
- Python 3.11+
- uv (for package management)
- Trained models: `random_forest_pipeline.joblib` and `catboost_model.cbm`

## Installation

```bash
# Install BentoML
uv pip install bentoml
```

## Setup Steps

### 1. Verify Model Files

Ensure your trained models are in the project directory:
- `random_forest_pipeline.joblib`
- `catboost_model.cbm`

### 2. Test Locally

Start the BentoML service locally:

```bash
uv run bentoml serve service:BankMarketingPredictor
```

The service will be available at:
- Main API: http://localhost:3000
- Swagger UI: http://localhost:3000/docs

### 4. Test the API

**Using cURL:**

```bash
# Predict with Random Forest
curl -X POST "http://localhost:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "random_forest",
    "data": {
      "age": 56,
      "job": "housemaid",
      "marital": "married",
      "education": "basic.4y",
      "default": "no",
      "housing": "no",
      "loan": "no",
      "contact": "telephone",
      "month": "may",
      "day_of_week": "mon",
      "duration": 261,
      "campaign": 1,
      "pdays": 999,
      "previous": 0,
      "poutcome": "nonexistent",
      "emp_var_rate": 1.1,
      "cons_price_idx": 93.994,
      "cons_conf_idx": -36.4,
      "euribor3m": 4.857,
      "nr_employed": 5191.0
    }
  }'

# Bulk prediction with CatBoost
curl -X POST "http://localhost:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "catboost",
    "data": [
      {
        "age": 56,
        "job": "housemaid",
        "marital": "married",
        "education": "basic.4y",
        "default": "no",
        "housing": "no",
        "loan": "no",
        "contact": "telephone",
        "month": "may",
        "day_of_week": "mon",
        "duration": 261,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0
      },
      {
        "age": 30,
        "job": "admin.",
        "marital": "single",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "fri",
        "duration": 180,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0
      }
    ]
  }'
```

**Using Python:**

```python
import bentoml

# Connect to local service
with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(
        model="random_forest",
        data={
            "age": 56,
            "job": "housemaid",
            "marital": "married",
            "education": "basic.4y",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "month": "may",
            "day_of_week": "mon",
            "duration": 261,
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp_var_rate": 1.1,
            "cons_price_idx": 93.994,
            "cons_conf_idx": -36.4,
            "euribor3m": 4.857,
            "nr_employed": 5191.0
        }
    )
    print(result)
```

## Available Endpoints

### 1. `/predict` (POST)
Unified endpoint that accepts model selection parameter:
- **model**: "random_forest" or "catboost"
- **data**: Single customer or list of customers

### 2. `/predict_random_forest` (POST)
Direct endpoint for Random Forest predictions:
- **data**: Single customer or list of customers

### 3. `/predict_catboost` (POST)
Direct endpoint for CatBoost predictions:
- **data**: Single customer or list of customers

### 4. `/health` (GET)
Health check endpoint

## Build Bento

Create a deployable Bento archive:

```bash
uv run bentoml build
```

This creates a versioned Bento in `~/bentoml/bentos/`.

## Deploy to BentoCloud

### Prerequisites
1. Create a BentoCloud account at https://cloud.bentoml.com
2. Get your API token from the dashboard

### Deployment Steps

```bash
# Login to BentoCloud
uv run bentoml cloud login

# Deploy the service
uv run bentoml deploy
```

The CLI will guide you through:
- Selecting or creating a cluster
- Configuring resources (CPU, memory, GPU if needed)
- Setting environment variables
- Configuring autoscaling

### Production Configuration

For production deployments, you can customize resources in `bentofile.yaml` or via deployment config:

```yaml
# deployment_config.yaml
services:
  BankMarketingPredictor:
    resources:
      cpu: "4"
      memory: "4Gi"
    scaling:
      min_replicas: 1
      max_replicas: 10
    envs:
      - name: LOG_LEVEL
        value: "INFO"
```

Deploy with config:
```bash
uv run bentoml deploy -f deployment_config.yaml
```

## Monitoring & Management

After deployment, use BentoCloud dashboard to:
- Monitor API requests and latency
- View model performance metrics
- Scale resources up/down
- Manage API keys
- View logs

## Docker Deployment (Alternative)

Build a Docker image:

```bash
uv run bentoml containerize bank_marketing_predictor:latest
```

Run the container:

```bash
docker run -p 3000:3000 bank_marketing_predictor:latest
```

## Advantages of BentoML

1. **Model Store**: Version control for ML models
2. **Performance**: Optimized for model serving with async support
3. **Scalability**: Easy horizontal scaling with BentoCloud
4. **Framework Agnostic**: Works with sklearn, CatBoost, PyTorch, TensorFlow, etc.
5. **Production Ready**: Built-in monitoring, logging, and observability
6. **Docker & Kubernetes**: Native containerization support

## Project Structure

```
mlbe/
├── service.py                      # BentoML service definition
├── bentofile.yaml                  # Bento configuration
├── save_models.py                  # Script to save models to Model Store
├── random_forest_pipeline.joblib   # Trained Random Forest model
├── catboost_model.cbm             # Trained CatBoost model
├── main.py                        # Original FastAPI implementation
└── BENTOML_SETUP.md              # This file
```

## Troubleshooting

### Models not loading
- Ensure `random_forest_pipeline.joblib` and `catboost_model.cbm` are in the same directory as `service.py`
- Check that custom transformers (CustomImputer, CyclicalFeatureTransformer) are defined in `service.py`

### Import errors
- Make sure all dependencies are in `bentofile.yaml`
- Custom transformers must be in `service.py` before the service class

### Port already in use
- Change port: `bentoml serve service:BankMarketingPredictor --port 3001`

## Next Steps

- Add model monitoring with BentoML's monitoring features
- Implement A/B testing between Random Forest and CatBoost
- Add request batching for improved throughput
- Set up CI/CD pipeline for automatic deployment
