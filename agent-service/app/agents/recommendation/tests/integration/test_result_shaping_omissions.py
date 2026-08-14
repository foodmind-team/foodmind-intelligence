from typing import Any

import httpx
from conftest import AGENT_FIXTURES, REPOSITORY_ROOT, load_json
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app

INFERENCE_FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


def _execute(request_name: str, inference_payload: dict[str, Any]) -> dict[str, Any]:
    raw = httpx.AsyncClient(
        base_url="http://inference.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=inference_payload)),
    )
    settings = Settings(app_env="test", internal_service_token=SecretStr("omission-token"))
    with TestClient(create_app(settings=settings, inference_http_client=raw)) as client:
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=load_json(AGENT_FIXTURES / request_name),
            headers={"Authorization": "Bearer omission-token"},
        )
    assert response.status_code == 200
    value = response.json()
    assert isinstance(value, dict)
    return value


def test_cold_start_omits_group_and_never_emits_cf_reasons() -> None:
    result = _execute("valid-cold-start.json", load_json(INFERENCE_FIXTURES / "valid-cold-start.json"))
    recommendations = result["recommendations"]
    assert isinstance(recommendations, list)
    assert [item["recommendationType"] for item in recommendations] == ["EXPLORATORY", "PERSONAL"]
    assert all(reason not in {"USER_CF", "ITEM_CF"} for item in recommendations for reason in item["reasons"])


def test_sparse_group_count_cannot_create_group_result() -> None:
    inference = {
        "contractVersion": "recommendation-inference-v1",
        "requestId": "req-sparse-group-001",
        "traceId": "trace-sparse-group-001",
        "status": "success",
        "modelVersion": "hybrid-ranking-v1",
        "modelPackageVersion": "recommendation-package-v1",
        "featureSchemaVersion": "recommendation-features-v2",
        "modelKeyVersion": "hmac-sha256-v1",
        "predictions": [
            {
                "candidateId": "candidate-sparse-a",
                "probability": 0.7,
                "modelScore": 0.85,
                "userCf": {"available": False, "score": None, "neighborSupport": 0},
                "itemCf": {"available": False, "score": None, "supportingItemCount": 0},
                "signals": {
                    "preferenceMatch": 0.7,
                    "wantToTry": False,
                    "groupPreferenceRate": 1.0,
                    "groupEligibleMemberCount": 1,
                    "contextMatch": 0.7,
                    "cleanlinessObserved": True,
                },
            }
        ],
    }
    result = _execute("valid-sparse-group.json", inference)
    recommendations = result["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) == 1
    assert recommendations[0]["recommendationType"] == "PERSONAL"
    assert "GROUP_POPULAR" not in recommendations[0]["reasons"]
