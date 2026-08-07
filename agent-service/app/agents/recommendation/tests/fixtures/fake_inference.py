"""Fixture-driven inference transport with no scoring or ML behavior."""

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from conftest import REPOSITORY_ROOT, load_json
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app

Scenario = Literal[
    "success",
    "timeout",
    "unavailable",
    "non_2xx",
    "malformed",
    "oversized",
    "feature_mismatch",
    "model_mismatch",
    "package_mismatch",
    "key_mismatch",
    "unknown_candidate",
    "duplicate_candidate",
    "missing_candidate",
    "invalid_probability",
    "invalid_evidence",
]
FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


@dataclass
class FixtureInference:
    scenario: Scenario = "success"
    attempts: int = 0
    last_request: dict[str, Any] | None = None
    last_response: dict[str, Any] | None = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        assert request.headers["authorization"].startswith("Bearer ")
        assert request.headers["x-inference-contract-version"] == "recommendation-inference-v1"
        body = json.loads(request.content)
        self.last_request = body
        assert request.headers["x-request-id"] == body["requestId"]
        assert request.headers["x-trace-id"] == body["traceId"]
        assert request.headers["x-feature-schema-version"] == body["featureSchemaVersion"]
        assert request.headers["x-model-key-version"] == body["modelKeyVersion"]
        assert body["contractVersion"] == "recommendation-inference-v1"
        assert body["deadlineAt"].endswith("Z")
        assert 1 <= len(body["candidates"]) <= 100
        if self.scenario == "timeout":
            raise httpx.ReadTimeout("fixture timeout", request=request)
        if self.scenario == "unavailable":
            return httpx.Response(503, content=b"fixture unavailable")
        if self.scenario == "non_2xx":
            return httpx.Response(500, content=b"fixture upstream body")
        if self.scenario == "malformed":
            return httpx.Response(200, content=b"{malformed")
        if self.scenario == "oversized":
            return httpx.Response(200, content=b"x" * 1025)

        payload = self._success_payload(body)
        predictions = payload["predictions"]
        if self.scenario == "feature_mismatch":
            payload["featureSchemaVersion"] = "recommendation-features-v1"
        elif self.scenario == "model_mismatch":
            payload["modelVersion"] = "hybrid-ranking-v0"
        elif self.scenario == "package_mismatch":
            payload["modelPackageVersion"] = "recommendation-package-v0"
        elif self.scenario == "key_mismatch":
            payload["modelKeyVersion"] = "hmac-sha256-v0"
        elif self.scenario == "unknown_candidate":
            predictions[0]["candidateId"] = "candidate-outside-request"
        elif self.scenario == "duplicate_candidate":
            predictions[1]["candidateId"] = predictions[0]["candidateId"]
        elif self.scenario == "missing_candidate":
            predictions.pop()
        elif self.scenario == "invalid_probability":
            predictions[0]["probability"] = 1.1
        elif self.scenario == "invalid_evidence":
            predictions[0]["userCf"]["available"] = False
        self.last_response = copy.deepcopy(payload)
        return httpx.Response(200, json=payload)

    def _success_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request["requestId"]
        if request_id == "req-cold-001":
            payload = copy.deepcopy(load_json(FIXTURES / "valid-cold-start.json"))
        elif request_id == "req-sparse-group-001":
            payload = _sparse_group_response()
        elif request_id.startswith("30000000-"):
            payload = _dynamic_success_response(request)
        else:
            payload = copy.deepcopy(load_json(FIXTURES / "valid-hybrid.json"))
        payload["requestId"] = request_id
        payload["traceId"] = request["traceId"]
        return payload


def _dynamic_success_response(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "contractVersion": "recommendation-inference-v1",
        "requestId": request["requestId"],
        "traceId": request["traceId"],
        "status": "success",
        "modelVersion": "hybrid-ranking-v1",
        "modelPackageVersion": "recommendation-package-v1",
        "featureSchemaVersion": "recommendation-features-v2",
        "modelKeyVersion": "hmac-sha256-v1",
        "predictions": [
            {
                "candidateId": candidate["candidateId"],
                "probability": 0.9 - index * 0.08,
                "modelScore": 2.0 - index * 0.25,
                "userCf": {"available": False, "score": None, "neighborSupport": 0},
                "itemCf": {"available": False, "score": None, "supportingItemCount": 0},
                "signals": candidate["evidence"],
            }
            for index, candidate in enumerate(request["candidates"])
        ],
    }


def _sparse_group_response() -> dict[str, Any]:
    return {
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


@contextmanager
def fixture_agent_client(
    scenario: Scenario = "success",
    *,
    inference_max_response_bytes: int = 262_144,
    enable_v1_compatibility: bool = False,
) -> Iterator[tuple[TestClient, FixtureInference]]:
    fake = FixtureInference(scenario)
    raw = httpx.AsyncClient(base_url="http://fixture-inference.test", transport=httpx.MockTransport(fake.handler))
    settings = Settings(
        app_env="test",
        internal_service_token=SecretStr("e2e-agent-token"),
        inference_service_token=SecretStr("e2e-inference-token"),
        inference_max_response_bytes=inference_max_response_bytes,
        enable_v1_compatibility=enable_v1_compatibility,
    )
    with TestClient(create_app(settings=settings, inference_http_client=raw)) as client:
        yield client, fake
