import hashlib
import json

import pytest
from conftest import REPOSITORY_ROOT
from pydantic import ValidationError

from recommendation_agent.schemas.inference_v1 import INFERENCE_RESPONSE_ADAPTER, InferenceFailure, InferenceSuccess

FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


def test_canonical_success_fixture_bytes_match_manifest_and_parse() -> None:
    manifest = json.loads((FIXTURES / "source-manifest.json").read_text(encoding="utf-8"))
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    for name in ("valid-hybrid.json", "valid-cold-start.json"):
        body = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == hashes[name]
        assert isinstance(INFERENCE_RESPONSE_ADAPTER.validate_json(body), InferenceSuccess)


def test_contract_failure_fixture_is_strict_but_compatibility_is_adapter_owned() -> None:
    failure = INFERENCE_RESPONSE_ADAPTER.validate_json((FIXTURES / "failure-package-incompatible.json").read_bytes())
    assert isinstance(failure, InferenceFailure)
    assert failure.error.code == "MODEL_PACKAGE_INCOMPATIBLE"


def test_wrong_feature_schema_fixture_fails_strict_success_model() -> None:
    with pytest.raises(ValidationError):
        INFERENCE_RESPONSE_ADAPTER.validate_json((FIXTURES / "invalid-feature-schema.json").read_bytes())
