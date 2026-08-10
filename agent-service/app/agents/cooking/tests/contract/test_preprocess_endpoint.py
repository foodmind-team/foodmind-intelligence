"""Contract tests for the internal cooking-plan preprocess endpoint.

POST /internal/v1/agents/cooking-plan/preprocess

The Spring Boot backend calls this endpoint BEFORE generate() so it can
reuse the agent's NL recipe parsing + gap-filling pipeline: raw recipe
text goes in, a fully-populated structured candidate comes out. The
backend then passes those candidates back as ``preparsed_candidates`` on
the generate request so the agent never re-parses or re-asks about gaps.

Validates:
  - Valid request returns populated candidates (200) with no unresolved
    critical gaps.
  - Missing/invalid service credential is rejected (401/422).
  - Extra request fields are rejected (422).
  - X-Request-ID is echoed in the response.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

_TEST_TOKEN = "test-internal-token-abc123"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"X-Internal-Token": _TEST_TOKEN}
    if extra:
        headers.update(extra)
    return headers


def _preprocess_body(overrides: dict | None = None) -> dict:
    """A recipe whose steps lack durations — the pipeline should infer them."""
    body = {
        "request_id": str(uuid.uuid4()),
        "recipes": (
            {
                "recipe_id": "r-tomato-eggs",
                "text": (
                    "Scrambled Eggs with Tomato\n"
                    "2 servings\n"
                    "Ingredients: 2 eggs, 1 tomato, salt, oil\n"
                    "Steps:\n"
                    "1. Heat oil in a pan. Whisk the eggs and season.\n"
                    "2. Add tomato and cook through. Serve hot.\n"
                ),
                "target_servings": "2",
            },
        ),
    }
    if overrides:
        body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# 1. Auth
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_token_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=_preprocess_body(),
        )
        assert response.status_code == 422

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=_preprocess_body(),
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert response.status_code == 401

    def test_valid_token_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=_preprocess_body(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_extra_field_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=_preprocess_body({"unknown_field": "nope"}),
            headers=_auth_headers(),
        )
        assert response.status_code == 422

    def test_missing_recipes_rejected(self, client: TestClient) -> None:
        body = _preprocess_body()
        del body["recipes"]
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=body,
            headers=_auth_headers(),
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. Behaviour — populated candidates, no unresolved critical gaps
# ---------------------------------------------------------------------------


class TestPreprocessBehaviour:
    def test_returns_filled_candidates_without_critical_gaps(self, client: TestClient) -> None:
        """The endpoint must fill missing data via local inference.

        Each returned candidate is a complete ExtractedRecipeCandidate whose
        inferable gaps (e.g. heat level) are already filled, so generate()
        can consume it without re-asking gap or assumption questions.
        """
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=_preprocess_body(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        body = response.json()
        assert "recipes" in body
        recipes = body["recipes"]
        assert len(recipes) == 1

        candidate = recipes[0]
        assert candidate["recipe_id"] == "r-tomato-eggs"
        assert candidate["dish_name"]
        # Ingredients and steps were extracted from the raw text.
        assert candidate["ingredients"]
        assert candidate["steps"]
        # The heating step's missing heat level was inferred (critical gap
        # resolved by local rules) — this is what lets generate() skip the
        # gap confirmation questions.
        heating_steps = [step for step in candidate["steps"] if "heat" in step["instruction"].lower()]
        assert heating_steps, "Expected a heating step to be extracted"
        assert heating_steps[0]["heat_level"] in {"LOW", "MEDIUM", "HIGH"}, (
            f"Heat level not inferred: {heating_steps[0].get('heat_level')}"
        )

    def test_request_id_echoed(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=_preprocess_body(),
            headers=_auth_headers({"X-Request-ID": "preprocess-req-1"}),
        )
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == "preprocess-req-1"

    def test_pan_fry_recipe_is_completed_without_empty_confirmation(self, client: TestClient) -> None:
        body = _preprocess_body()
        body["recipes"] = (
            {
                "recipe_id": "qa-ginger-tofu",
                "text": (
                    "Ginger Tofu Bowl\n"
                    "4 servings\n"
                    "Ingredients: 400 g tofu, 20 g ginger, 1 tbsp oil\n"
                    "Steps:\n"
                    "1. Slice the tofu and ginger, then pan-fry until the tofu is golden.\n"
                ),
                "target_servings": "4",
            },
        )

        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=body,
            headers=_auth_headers(),
        )

        assert response.status_code == 200
        step = response.json()["recipes"][0]["steps"][0]
        assert step["category"] == "heating"
        assert step["heat_level"] == "MEDIUM"
        assert step["active_duration_minutes"] is not None or step["passive_duration_minutes"] is not None
        assert step["resources_hint"]

    def test_repeated_heat_language_gets_a_bounded_duration(self, client: TestClient) -> None:
        """Regression: real shrimp imports often say heat/fry-until-golden.

        The generic heat technique previously had no duration fallback, so
        planning stopped at NEEDS_CONFIRMATION even after preprocessing.
        """
        body = _preprocess_body()
        body["recipes"] = (
            {
                "recipe_id": "qa-shrimp",
                "text": (
                    "Crispy Shrimp\n"
                    "2 servings\n"
                    "Ingredients: 300 g shrimp, 2 tbsp oil, 10 g garlic\n"
                    "Steps:\n"
                    "1. Heat oil in a pan, add the shrimp, and fry until golden and crispy.\n"
                    "2. Leave some oil in the pan, heat it, add garlic, and stir-fry until fragrant.\n"
                ),
                "target_servings": "2",
            },
        )

        response = client.post(
            "/internal/v1/agents/cooking-plan/preprocess",
            json=body,
            headers=_auth_headers(),
        )

        assert response.status_code == 200
        heating_steps = [step for step in response.json()["recipes"][0]["steps"] if step["category"] == "heating"]
        assert heating_steps
        assert all(
            step["active_duration_minutes"] is not None or step["passive_duration_minutes"] is not None
            for step in heating_steps
        )
