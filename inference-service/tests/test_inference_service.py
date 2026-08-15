from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model import FEATURE_NAMES


def _package(directory: Path) -> None:
    artifact = directory / "hybrid_lr_model.npz"
    np.savez(
        artifact,
        weights=np.linspace(0.0, 1.0, len(FEATURE_NAMES) + 1),
        mean=np.zeros(len(FEATURE_NAMES)),
        std=np.ones(len(FEATURE_NAMES)),
        feature_names=np.array(FEATURE_NAMES),
    )
    manifest = {
        "packageVersion": "recommendation-package-v1",
        "modelVersion": "hybrid-ranking-v1",
        "featureSchemaVersion": "recommendation-features-v2",
        "inferenceContractVersion": "recommendation-inference-v1",
        "modelKeyVersion": "hmac-sha256-v1",
        "artifact": artifact.name,
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "featureNames": list(FEATURE_NAMES),
        "approvedFor": ["local"],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _request() -> dict[str, object]:
    return {
        "contractVersion": "recommendation-inference-v1",
        "requestId": "request-1",
        "traceId": "trace-1",
        "deadlineAt": (datetime.now(UTC) + timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
        "featureSchemaVersion": "recommendation-features-v2",
        "modelUserKey": "A" * 43,
        "modelKeyVersion": "hmac-sha256-v1",
        "candidates": [
            {
                "candidateId": "candidate-1",
                "modelMealKey": "meal_key_00000001",
                "modelOfferingKey": "offering_key_0001",
                "evidence": {
                    "preferenceMatch": 0.9,
                    "wantToTry": True,
                    "groupPreferenceRate": None,
                    "groupEligibleMemberCount": 0,
                    "contextMatch": 0.8,
                    "cleanlinessObserved": True,
                },
            }
        ],
    }


def test_loads_package_and_scores_candidates(tmp_path: Path) -> None:
    _package(tmp_path)
    settings = Settings(model_package_dir=tmp_path, internal_service_token="test-inference-token")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post(
            "/internal/v1/recommendations/score",
            headers={"Authorization": "Bearer test-inference-token"},
            json=_request(),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["predictions"][0]["candidateId"] == "candidate-1"
    assert 0.0 <= body["predictions"][0]["probability"] <= 1.0
    assert body["predictions"][0]["signals"] == _request()["candidates"][0]["evidence"]
    assert body["predictions"][0]["userCf"] == {"available": False, "score": None, "neighborSupport": 0}
    assert body["predictions"][0]["itemCf"] == {"available": False, "score": None, "supportingItemCount": 0}


def test_exposes_verified_collaborative_signals(tmp_path: Path) -> None:
    _package(tmp_path)
    index = {
        "schemaVersion": "foodmind-collaborative-index-v1",
        "sourceSnapshotSha256": "a" * 64,
        "positiveOnly": True,
        "userCf": {"A" * 43: {"meal_key_00000001": {"score": 0.8, "support": 3}}},
        "itemCf": {"A" * 43: {"meal_key_00000001": {"score": 0.7, "support": 2}}},
    }
    raw = json.dumps(index).encode("utf-8")
    (tmp_path / "collaborative_index.json").write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collaborativeIndex"] = {
        "artifact": "collaborative_index.json",
        "artifactSha256": hashlib.sha256(raw).hexdigest(),
        "schemaVersion": "foodmind-collaborative-index-v1",
        "sourceSnapshotSha256": "a" * 64,
        "positiveOnly": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    settings = Settings(model_package_dir=tmp_path, internal_service_token="test-inference-token")
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/internal/v1/recommendations/score",
            headers={"Authorization": "Bearer test-inference-token"},
            json=_request(),
        )
    prediction = response.json()["predictions"][0]
    assert prediction["userCf"] == {"available": True, "score": 0.8, "neighborSupport": 3}
    assert prediction["itemCf"] == {"available": True, "score": 0.7, "supportingItemCount": 2}


def test_requires_service_authentication(tmp_path: Path) -> None:
    _package(tmp_path)
    with TestClient(create_app(Settings(model_package_dir=tmp_path))) as client:
        assert client.post("/internal/v1/recommendations/score", json=_request()).status_code == 401


def test_checksum_failure_keeps_liveness_but_fails_readiness(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "hybrid_lr_model.npz").write_bytes(b"tampered")
    with TestClient(create_app(Settings(model_package_dir=tmp_path))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
