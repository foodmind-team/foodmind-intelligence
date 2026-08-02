"""P3-04: policy resolution inside validate_safety_node and terminal responses.

Verifies:
  - unknown region → SAFETY_POLICY_UNAVAILABLE error (never silent fallback)
  - request region overrides the deployment default
  - READY / CONFIRMATION responses record the policy provenance
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.models import (
    GeneratePlanRequest,
    RecipeInput,
)
from cooking_plan_agent.rendering.responses import (
    render_confirmation_response,
    render_ready_response,
)
from cooking_plan_agent.workflow.nodes import validate_safety_node


class _FakeRuntime:
    """Minimal runtime stand-in — nodes only need .context at call time."""

    def __init__(self, context: object) -> None:
        self.context = context


def _ctx(safety_engine=None) -> object:
    return type("C", (), {"safety_engine": safety_engine})()


def _request(region: str | None = None) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="req-policy",
        user_id="u",
        recipes=(RecipeInput(recipe_id="r1", text="Cook.", target_servings=Decimal(2)),),
        region=region,
    )


@pytest.mark.asyncio
async def test_unknown_region_is_rejected_with_stable_code() -> None:
    """An explicit unknown region must FAIL — never silently fall back (D6)."""
    state = {"request": _request(region="XX")}
    result = await validate_safety_node(state, _FakeRuntime(_ctx()))

    error = result.get("error")
    assert error is not None
    assert error.error_code == "SAFETY_POLICY_UNAVAILABLE"
    assert "safety_report" not in result


@pytest.mark.asyncio
async def test_request_region_overrides_deployment_default(monkeypatch) -> None:
    """request.region='SG' wins over COOKING_PLAN_SAFETY_POLICY_REGION=US."""
    monkeypatch.setenv("COOKING_PLAN_SAFETY_POLICY_REGION", "US")
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    try:
        state = {"request": _request(region="SG")}
        result = await validate_safety_node(state, _FakeRuntime(_ctx()))
    finally:
        get_settings.cache_clear()

    record = result.get("safety_policy")
    assert record is not None
    assert record.region == "SG"
    assert record.version == "1.0"
    assert record.sources, "SG policy must carry official sources"


@pytest.mark.asyncio
async def test_default_region_applies_when_request_omits_region() -> None:
    """No request region → the deployment default (US) applies."""
    state = {"request": _request(region=None)}
    result = await validate_safety_node(state, _FakeRuntime(_ctx()))

    record = result.get("safety_policy")
    assert record is not None
    assert record.region == "US"
    # No error: the resolved US pack evaluates the (empty) recipe set safely.
    assert "error" not in result
    report = result.get("safety_report")
    assert report is not None and report.is_safe is True


@pytest.mark.asyncio
async def test_sg_report_and_policy_both_recorded() -> None:
    """SafetyReport and state both carry the SG provenance."""
    state = {"request": _request(region="SG")}
    result = await validate_safety_node(state, _FakeRuntime(_ctx()))

    report = result["safety_report"]
    assert report.safety_policy is not None
    assert report.safety_policy.region == "SG"
    assert result["safety_policy"].region == "SG"


# ---------------------------------------------------------------------------
# Terminal responses record policy provenance
# ---------------------------------------------------------------------------


def test_ready_response_records_policy() -> None:
    from cooking_plan_agent.domain.enums import SolverStatus
    from cooking_plan_agent.safety.policy import resolve_policy
    from cooking_plan_agent.scheduling.models import ScheduleResult

    record = resolve_policy("US").to_record()
    state = {
        "request": _request(),
        "safety_policy": record,
        "schedule_result": ScheduleResult(status=SolverStatus.OPTIMAL, makespan_minutes=30),
    }
    response = render_ready_response(state)
    assert response.safety_policy is not None
    assert response.safety_policy.region == "US"
    assert response.safety_policy.version == "1.0"


def test_confirmation_response_records_policy() -> None:
    from cooking_plan_agent.safety.policy import resolve_policy

    record = resolve_policy("SG").to_record()
    state = {
        "request": _request(),
        "safety_policy": record,
        "parsed_recipes": (),
        "repair_options": (),
    }
    response = render_confirmation_response(state)
    assert response.safety_policy is not None
    assert response.safety_policy.region == "SG"
