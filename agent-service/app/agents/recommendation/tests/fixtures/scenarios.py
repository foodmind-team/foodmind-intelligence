"""Independent Backend-style fixture validation and fallback harness."""

import json
import math
from pathlib import Path
from typing import Any

from conftest import REPOSITORY_ROOT
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

_RESPONSE_SCHEMA = json.loads(
    (REPOSITORY_ROOT / "contracts/internal/agent/recommendation/v2/response.schema.json").read_text(encoding="utf-8")
)
_VALIDATOR = Draft202012Validator(_RESPONSE_SCHEMA, format_checker=FormatChecker())
FALLBACK = {"status": "fallback", "recommendations": [], "reason": "INTELLIGENCE_UNAVAILABLE"}
_TEMPLATES = {
    "USER_CF": "People with similar preferences also liked this.",
    "ITEM_CF": "It resembles meals you liked.",
    "PREFERENCE_MATCH": "It matches your saved preferences.",
    "WANT_TO_TRY": "You marked this as Want to Try.",
    "GROUP_POPULAR": "It matches preferences shared by the group.",
    "CONTEXT_MATCH": "It matches the current meal context.",
    "CLEANLINESS_OBSERVED": "A recent cleanliness observation is recorded.",
}


def backend_validate(
    request: dict[str, Any],
    inference: dict[str, Any],
    response: dict[str, Any],
) -> bool:
    if list(_VALIDATOR.iter_errors(response)):
        return False
    recommendations = response["recommendations"]
    types = [item["recommendationType"] for item in recommendations]
    type_order = {"PERSONAL": 0, "EXPLORATORY": 1, "GROUP_INSPIRED": 2}
    if len(types) != len(set(types)) or types != sorted(types, key=type_order.__getitem__):
        return False
    request_ids = {item["candidateId"] for item in request["candidates"]}
    if not {item["candidateId"] for item in recommendations}.issubset(request_ids):
        return False
    predictions = {item["candidateId"]: item for item in inference["predictions"]}
    evidence = {item["candidateId"]: item["evidence"] for item in request["candidates"]}
    for item in recommendations:
        prediction = predictions.get(item["candidateId"])
        facts = evidence.get(item["candidateId"])
        if prediction is None or facts is None:
            return False
        if not math.isclose(item["probability"], prediction["probability"], rel_tol=0.0, abs_tol=0.0):
            return False
        if not math.isclose(item["modelScore"], prediction["modelScore"], rel_tol=0.0, abs_tol=0.0):
            return False
        if not _reasons_supported(item["reasons"], prediction, facts):
            return False
        if item["explanation"] != " ".join(_TEMPLATES[reason] for reason in item["reasons"]):
            return False
        if item["recommendationType"] == "PERSONAL" and not _personal_supported(prediction, facts):
            return False
        if item["recommendationType"] == "GROUP_INSPIRED" and not _group_supported(facts):
            return False
    return True


def _personal_supported(prediction: dict[str, Any], facts: dict[str, Any]) -> bool:
    user_cf = prediction["userCf"]
    return bool(
        (user_cf["available"] and user_cf["score"] >= 0.6 and user_cf["neighborSupport"] >= 3)
        or facts["preferenceMatch"] >= 0.7
    )


def _group_supported(facts: dict[str, Any]) -> bool:
    return bool(
        facts["groupPreferenceRate"] is not None
        and facts["groupPreferenceRate"] >= 0.6
        and facts["groupEligibleMemberCount"] >= 2
    )


def _reasons_supported(reasons: list[str], prediction: dict[str, Any], facts: dict[str, Any]) -> bool:
    user_cf = prediction["userCf"]
    item_cf = prediction["itemCf"]
    supported = {
        "USER_CF": user_cf["available"] and user_cf["score"] >= 0.6 and user_cf["neighborSupport"] >= 3,
        "ITEM_CF": item_cf["available"] and item_cf["score"] >= 0.6 and item_cf["supportingItemCount"] >= 2,
        "PREFERENCE_MATCH": facts["preferenceMatch"] >= 0.7,
        "WANT_TO_TRY": facts["wantToTry"] is True,
        "GROUP_POPULAR": facts["groupPreferenceRate"] is not None
        and facts["groupPreferenceRate"] >= 0.6
        and facts["groupEligibleMemberCount"] >= 2,
        "CONTEXT_MATCH": facts["contextMatch"] is not None and facts["contextMatch"] >= 0.7,
        "CLEANLINESS_OBSERVED": facts["cleanlinessObserved"] is True,
    }
    return bool(reasons) and all(supported.get(reason, False) for reason in reasons)


def backend_call(client: TestClient, request: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        "/internal/v1/recommendations/generate",
        json=request,
        headers={"Authorization": "Bearer e2e-agent-token"},
    )
    if response.status_code != 200:
        return dict(FALLBACK)
    body = response.json()
    return body if isinstance(body, dict) else dict(FALLBACK)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
