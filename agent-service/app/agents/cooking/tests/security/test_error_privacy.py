"""Security tests for P2-03 — error message privacy.

Asserts that FAILED responses never echo internal exception text, provider
payloads or secrets, and that failures remain traceable via correlation ID.
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


class _FakeRecipeExtractor:
    """Returns a gap-free candidate so the workflow reaches solve_schedule."""

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        return ExtractedRecipeCandidate(
            recipe_id="test-recipe-1",
            dish_name="Test Dish",
            original_servings=2,
            source_language="en",
            ingredients=(
                ExtractedIngredient(
                    raw_text="chicken 200g",
                    name="chicken breast",
                    quantity=200,
                    unit="g",
                ),
            ),
            steps=(
                ExtractedStep(
                    step_number=1,
                    instruction="Cook for 10 minutes",
                    category="heating",
                    active_duration_minutes=10,
                    heat_level=HeatLevel.HIGH,
                    target_temperature_c=Decimal(200),
                    resources_hint=("stove",),
                ),
            ),
        )


@pytest.fixture
def graph():
    return build_cooking_plan_graph()


@pytest.fixture
def context():
    return WorkflowContext(recipe_extractor=_FakeRecipeExtractor())


@pytest.fixture
def valid_request():
    return GeneratePlanRequest(
        request_id="privacy-req-001",
        user_id="user-1",
        recipes=(
            {
                "recipe_id": "r1",
                "text": "Cook chicken for 10 minutes. Serves 2.",
                "target_servings": 2,
            },
        ),
        inventory_lots=(
            InventoryLotSnapshot(
                lot_id="lot-001",
                item_id="item-001",
                canonical_name="chicken breast",
                on_hand=Decimal(300),
                reserved=Decimal(0),
                unit="g",
            ),
        ),
        kitchen_resources=(
            KitchenResourceSnapshot(
                resource_id="stove-1",
                resource_type="stove",
                capacity=Decimal(4),
                capacity_unit="burners",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_provider_secret_never_reaches_response(
    graph,
    context,
    valid_request,
    monkeypatch,
):
    """A provider exception containing a live API key must not leak.

    The exception is raised inside solve_schedule_node, which now retains
    only the exception type as diagnostics (P2-03) — the secret stays out of
    the state, the log and the response.
    """
    import cooking_plan_agent.scheduling.orchestrator as orchestrator_module

    secret = "sk-live-abcdef1234567890"

    class _ExplodingOrchestrator:
        def solve(self, problem, level):  # noqa: ANN001, ANN002
            raise RuntimeError(f"provider returned 401 body with api_key={secret}")

    monkeypatch.setattr(orchestrator_module, "ScheduleOrchestrator", _ExplodingOrchestrator)

    result = await graph.ainvoke(
        {"request": valid_request},
        context=context,
        config={"recursion_limit": 30},
    )

    response = result["response"]
    assert response.status == "FAILED"
    assert response.error_code == "INTERNAL_ERROR"
    assert secret not in response.message
    assert response.correlation_id == valid_request.request_id

    error = result.get("error")
    assert error is not None
    assert error.diagnostics == {"exception_type": "RuntimeError"}


@pytest.mark.asyncio
async def test_node_message_with_secret_is_sanitised(
    graph,
    context,
    valid_request,
    monkeypatch,
):
    """Even if a node writes a secret into its internal message, the response
    must resolve the public text from the message catalog (P2-03)."""
    import cooking_plan_agent.workflow.graph as graph_module

    secret = "Bearer token=eyJhbGciOiJIUzI1NiJ9.recipe"

    async def _leaky_solve(state, runtime):  # noqa: ANN001
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_UNKNOWN",
                message=f"raw provider detail {secret}",
                correlation_id=valid_request.request_id,
                node_name="solve_schedule",
            )
        }

    monkeypatch.setattr(graph_module, "solve_schedule_node", _leaky_solve)
    graph2 = build_cooking_plan_graph()

    result = await graph2.ainvoke(
        {"request": valid_request},
        context=context,
        config={"recursion_limit": 30},
    )

    response = result["response"]
    assert response.status == "FAILED"
    assert response.error_code == "SCHEDULE_UNKNOWN"
    assert secret not in response.message
    assert response.correlation_id == valid_request.request_id
    # Public text comes from the catalog row, not the raw message.
    assert "time limit" in response.message


@pytest.mark.asyncio
async def test_failed_diagnostic_log_is_traceable(
    graph,
    context,
    valid_request,
    monkeypatch,
    caplog,
):
    """The FAILED log line carries error code + correlation ID so the
    incident can be traced without writing the raw message (P2-03)."""
    import cooking_plan_agent.workflow.graph as graph_module

    async def _failing_solve(state, runtime):  # noqa: ANN001
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message="sensitive internal detail",
                correlation_id=valid_request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": "ValueError"},
            )
        }

    monkeypatch.setattr(graph_module, "solve_schedule_node", _failing_solve)
    graph2 = build_cooking_plan_graph()

    with caplog.at_level("WARNING", logger="cooking_plan_agent.workflow.nodes"):
        result = await graph2.ainvoke(
            {"request": valid_request},
            context=context,
            config={"recursion_limit": 30},
        )

    assert result["response"].status == "FAILED"
    log_text = caplog.text
    assert "SCHEDULE_VERIFICATION_FAILED" in log_text
    assert valid_request.request_id in log_text
    assert "sensitive internal detail" not in log_text
