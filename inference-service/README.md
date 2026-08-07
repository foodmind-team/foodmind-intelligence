# FoodMind Recommendation Inference

Private FastAPI runtime for `recommendation-inference-v1`. It validates and
loads an immutable model package produced by `foodmind-ml`, verifies the model
artifact checksum, scores the already-authorised candidates supplied by the
Recommendation Agent, and returns model/version metadata.

## Local run

Build the local-only model package from the sibling ML repository, then:

```powershell
uv sync --frozen --dev
$env:INFERENCE_MODEL_PACKAGE_DIR='..\..\.tmp\runtime\model-package'
$env:INFERENCE_INTERNAL_SERVICE_TOKEN='local-inference-only'
uv run uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Health endpoints are `GET /health/live` and `GET /health/ready`. The private
scoring endpoint is `POST /internal/v1/recommendations/score` and requires a
Bearer service token. This service never queries the FoodMind database and does
not apply hard eligibility rules.
