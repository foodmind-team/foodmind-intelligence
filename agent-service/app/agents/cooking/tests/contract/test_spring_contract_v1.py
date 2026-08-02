"""Spring Boot v1 contract tests (P0-02).

Verifies the compat endpoint against the REAL Java DTO shape
(AgentCookingRequest / AgentCookingResponse) and the
CookingPlanResultValidator equivalent rules:

  - Bearer auth (Java sends ``Authorization: Bearer <token>``)
  - camelCase request/response serialisation with no extra fields
  - contract version / requestId / planId / traceId echo-back
  - READY → SUCCEEDED mapping with candidate-selected sourceRecipeId
  - non-READY terminal states → FAILED
  - deadlineAt fast-fail and execution budget
  - unknown candidate ID / wrong token / unsupported version
  - performance gate: compat request without external providers < 700ms
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

_TEST_TOKEN = "test-internal-token-abc123"
_CONTRACT_VERSION = "cooking-agent-v1"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Purge the settings lru_cache so the monkeypatched token is actually
    # read on the next get_settings() call (create_app now reads Settings
    # to configure CORS).
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)
    monkeypatch.setenv("COOKING_PLAN_LLM_ENABLED", "false")


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------------------
# Fixture builders — mirror the Java contract fixture (valid-normal-response)
# ---------------------------------------------------------------------------


def _candidate(
    recipe_id: str | None = None,
    name: str = "Tofu Bowl",
    with_steps: bool = True,
) -> dict:
    """Build a candidate snapshot equivalent to RecipeCandidate serialisation."""
    snapshot = {
        "recipeId": recipe_id or str(uuid.uuid4()),
        "name": name,
        "description": "A simple tofu rice bowl.",
        "defaultServings": 2,
        "totalMinutes": 35,
        "estimatedCost": "9.50",
        "currency": "SGD",
        "dietaryTagCodes": ["VEGETARIAN"],
        "allergenCodes": ["SOY"],
        "ingredients": (
            {
                "sequenceNo": 1,
                "ingredientName": "Firm tofu",
                "quantity": "300",
                "unit": "g",
                "optional": False,
            },
        ),
        "steps": (
            {"stepNo": 1, "instruction": "Pan-fry the firm tofu until golden."},
            {"stepNo": 2, "instruction": "Serve the tofu over cooked rice."},
        )
        if with_steps
        else (),
    }
    return {"recipeId": snapshot["recipeId"], "snapshot": snapshot}


def _compat_body(overrides: dict | None = None) -> dict:
    """Build a full AgentCookingRequest-compatible payload."""
    body = {
        "contractVersion": _CONTRACT_VERSION,
        "requestId": str(uuid.uuid4()),
        "planId": str(uuid.uuid4()),
        "traceId": "trace-cooking-normal",
        "deadlineAt": (datetime.now(UTC) + timedelta(seconds=10)).isoformat(),
        "request": {
            "contractVersion": "cooking-public-v1",
            "ingredients": ({"ingredientName": "Firm tofu", "quantity": "300", "unit": "g", "source": "MANUAL"},),
            "servings": 2,
            "maxMinutes": 60,
            "maxBudget": "20.00",
            "currency": "SGD",
            "constraints": {
                "requiredDietaryTagCodes": ["VEGETARIAN"],
                "avoidAllergenCodes": ["PEANUT"],
            },
        },
        "preferences": {
            "requiredDietaryTagCodes": ["VEGETARIAN"],
            "avoidAllergenCodes": ["PEANUT"],
        },
        "candidates": (_candidate(),),
    }
    if overrides:
        body.update(overrides)
    return body


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {_TEST_TOKEN}"}
    if extra:
        headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# 1. Authentication — Bearer credential
# ---------------------------------------------------------------------------


class TestBearerAuth:
    def test_missing_auth_returns_401(self, client: TestClient) -> None:
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body())
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "MISSING_AUTHORIZATION_HEADER"

    def test_wrong_token_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json=_compat_body(),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "INVALID_INTERNAL_CREDENTIAL"

    def test_non_bearer_scheme_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json=_compat_body(),
            headers={"Authorization": f"Basic {_TEST_TOKEN}"},
        )
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "INVALID_AUTHORIZATION_SCHEME"

    def test_valid_bearer_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json=_compat_body(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "SUCCEEDED"


# ---------------------------------------------------------------------------
# 2. Envelope echo-back and schema strictness
# ---------------------------------------------------------------------------


class TestEnvelopeAndSchema:
    def test_request_id_plan_id_trace_id_echoed(self, client: TestClient) -> None:
        body = _compat_body()
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["contractVersion"] == _CONTRACT_VERSION
        assert payload["requestId"] == body["requestId"]
        assert payload["planId"] == body["planId"]
        assert payload["traceId"] == body["traceId"]

    def test_unknown_field_rejected(self, client: TestClient) -> None:
        """Java's fail-on-unknown-properties=true → Python side extra=forbid."""
        body = _compat_body({"unexpectedField": "boom"})
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 422

    def test_response_has_no_extra_fields(self, client: TestClient) -> None:
        """The response must serialize ONLY the record fields Java knows."""
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body(), headers=_auth_headers())
        payload = response.json()
        allowed = {
            "contractVersion",
            "requestId",
            "planId",
            "traceId",
            "agentTraceId",
            "status",
            "sourceRecipeId",
            "servings",
            "totalMinutes",
            "estimatedCost",
            "currency",
            "ingredients",
            "steps",
            "warnings",
        }
        assert set(payload.keys()) <= allowed

    def test_unsupported_contract_version_fast_fail(self, client: TestClient) -> None:
        body = _compat_body({"contractVersion": "cooking-agent-v0"})
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 200
        assert response.json()["status"] == "FAILED"


# ---------------------------------------------------------------------------
# 3. Validator-equivalent rules on SUCCEEDED responses
# ---------------------------------------------------------------------------


class TestValidatorEquivalents:
    def test_servings_matches_request(self, client: TestClient) -> None:
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body(), headers=_auth_headers())
        payload = response.json()
        assert payload["status"] == "SUCCEEDED"
        assert payload["servings"] == 2  # request.request.servings

    def test_source_recipe_id_in_candidates(self, client: TestClient) -> None:
        body = _compat_body()
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        payload = response.json()
        candidate_ids = {c["recipeId"] for c in body["candidates"]}
        assert payload["sourceRecipeId"] in candidate_ids

    def test_ingredients_steps_contiguous_and_non_empty(self, client: TestClient) -> None:
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body(), headers=_auth_headers())
        payload = response.json()
        ingredients = payload["ingredients"]
        steps = payload["steps"]
        assert len(ingredients) > 0
        assert len(steps) > 0
        # sequenceNo/stepNo must be 1..N contiguous (validator requireContiguous)
        assert [i["sequenceNo"] for i in ingredients] == list(range(1, len(ingredients) + 1))
        assert [s["stepNo"] for s in steps] == list(range(1, len(steps) + 1))

    def test_ingredient_availability_and_quantity_rules(self, client: TestClient) -> None:
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body(), headers=_auth_headers())
        payload = response.json()
        for ing in payload["ingredients"]:
            assert ing["availability"] in ("AVAILABLE", "TO_BUY")
            # quantity and unit must be both set or both null
            assert (ing["quantity"] is None) == (ing["unit"] is None)

    def test_warning_codes_in_allow_list(self, client: TestClient) -> None:
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body(), headers=_auth_headers())
        payload = response.json()
        allowed = {
            "CHECK_ALLERGEN_LABELS",
            "MAY_REQUIRE_EXTRA_TIME",
            "BUDGET_ESTIMATE_ONLY",
            "PANTRY_ITEM_UNVERIFIED",
            "COOK_THOROUGHLY",
        }
        for w in payload["warnings"]:
            assert w["warningCode"] in allowed


# ---------------------------------------------------------------------------
# 4. Deadline semantics
# ---------------------------------------------------------------------------


class TestDeadline:
    def test_past_deadline_fast_fails(self, client: TestClient) -> None:
        body = _compat_body({"deadlineAt": (datetime.now(UTC) - timedelta(seconds=5)).isoformat()})
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 200
        assert response.json()["status"] == "FAILED"


# ---------------------------------------------------------------------------
# 5. Non-READY terminal states → FAILED
# ---------------------------------------------------------------------------


class TestFailureMapping:
    def test_empty_candidates_fails_gracefully(self, client: TestClient) -> None:
        body = _compat_body({"candidates": ()})
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "FAILED"
        # Envelope still echoed for triage
        assert payload["requestId"] == body["requestId"]

    def test_candidate_without_usable_content_fails(self, client: TestClient) -> None:
        candidate = _candidate(with_steps=False)
        candidate["snapshot"]["ingredients"] = ()
        body = _compat_body({"candidates": (candidate,)})
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 200
        assert response.json()["status"] == "FAILED"


# ---------------------------------------------------------------------------
# 6. Performance gate — no external provider, < 700ms
# ---------------------------------------------------------------------------


class TestPerformanceGate:
    @pytest.mark.slow
    def test_compat_request_within_700ms(self, client: TestClient) -> None:
        """P0-02: provider-free compat request must complete within 700ms."""
        start = time.monotonic()
        response = client.post("/internal/v1/cooking-plans/generate", json=_compat_body(), headers=_auth_headers())
        elapsed_ms = (time.monotonic() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 700, f"Compat request took {elapsed_ms:.1f}ms (> 700ms budget)"


# ---------------------------------------------------------------------------
# 7. Serialisation fidelity — Java fixture equivalent
# ---------------------------------------------------------------------------


class TestJavaFixtureEquivalence:
    def test_valid_normal_response_fixture(self, client: TestClient) -> None:
        """Reproduce the Java contract fixture valid-normal-response.json."""
        body = {
            "contractVersion": _CONTRACT_VERSION,
            "requestId": "aaaaaaaa-0000-4000-8000-000000000001",
            "planId": "bbbbbbbb-0000-4000-8000-000000000001",
            "traceId": "trace-cooking-normal",
            "deadlineAt": (datetime.now(UTC) + timedelta(seconds=10)).isoformat(),
            "request": {
                "contractVersion": "cooking-public-v1",
                "ingredients": ({"ingredientName": "Firm tofu", "quantity": "300", "unit": "g", "source": "MANUAL"},),
                "servings": 2,
                "maxMinutes": 60,
                "maxBudget": "20.00",
                "currency": "SGD",
                "constraints": {"requiredDietaryTagCodes": ["VEGETARIAN"], "avoidAllergenCodes": ["PEANUT"]},
            },
            "preferences": {"requiredDietaryTagCodes": ["VEGETARIAN"], "avoidAllergenCodes": ["PEANUT"]},
            "candidates": (
                {
                    "recipeId": "30000000-0000-4000-8000-000000000001",
                    "snapshot": {
                        "recipeId": "30000000-0000-4000-8000-000000000001",
                        "name": "Tofu Rice Bowl",
                        "description": "A simple tofu rice bowl.",
                        "defaultServings": 2,
                        "totalMinutes": 35,
                        "estimatedCost": "9.50",
                        "currency": "SGD",
                        "dietaryTagCodes": ["VEGETARIAN"],
                        "allergenCodes": ["SOY"],
                        "ingredients": (
                            {
                                "sequenceNo": 1,
                                "ingredientName": "Firm tofu",
                                "quantity": "300",
                                "unit": "g",
                                "optional": False,
                            },
                        ),
                        "steps": (
                            {"stepNo": 1, "instruction": "Cook the jasmine rice according to its package directions."},
                        ),
                    },
                },
            ),
        }
        response = client.post("/internal/v1/cooking-plans/generate", json=body, headers=_auth_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "SUCCEEDED"
        assert payload["contractVersion"] == _CONTRACT_VERSION
        assert payload["requestId"] == body["requestId"]
        assert payload["planId"] == body["planId"]
        assert payload["traceId"] == body["traceId"]
        assert payload["sourceRecipeId"] == "30000000-0000-4000-8000-000000000001"
        assert payload["servings"] == 2
        assert payload["totalMinutes"] is not None and payload["totalMinutes"] > 0
        assert payload["ingredients"], "SUCCEEDED must include at least one ingredient"
        assert payload["steps"], "SUCCEEDED must include at least one step"
