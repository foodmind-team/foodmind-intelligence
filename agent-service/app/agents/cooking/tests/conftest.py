"""Root conftest — shared fixtures and plugins for all test categories.

Handbook 11.2: tests must be deterministic and offline.
No external keys, no live websites, no nondeterministic model output.
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    KitchenResourceSnapshot,
    ResourceNeed,
    TaskDependency,
)

# =============================================================================
# Domain object factories — used across unit/integration/contract tests
# =============================================================================


def _task(
    task_id: str,
    dish_id: str = "d1",
    duration: int = 5,
    work_mode: WorkMode = WorkMode.ACTIVE,
    category: str = "general",
    deps: tuple[TaskDependency, ...] = (),
    resources: tuple[ResourceNeed, ...] = (),
) -> CookingTask:
    """Create a minimal CookingTask — shared across scheduling and preparation tests."""
    return CookingTask(
        task_id=task_id,
        dish_id=dish_id,
        instruction=f"Task {task_id}",
        duration_minutes=duration,
        work_mode=work_mode,
        category=category,
        dependencies=deps,
        resources=resources,
    )


def _stove(count: int = 4) -> KitchenResourceSnapshot:
    """Create a stove resource with N burners."""
    return KitchenResourceSnapshot(
        resource_id="stove:main",
        resource_type="stove",
        capacity=Decimal(count),
        capacity_unit="burners",
    )


def _oven() -> KitchenResourceSnapshot:
    """Create an oven resource."""
    return KitchenResourceSnapshot(
        resource_id="oven:main",
        resource_type="oven",
        capacity=Decimal(1),
    )


# =============================================================================
# Plugin registration — Hypothesis settings for CI determinism
# =============================================================================


def pytest_configure(config):
    """Register custom markers and Hypothesis profile."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take > 1 second (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "golden: solver golden tests with known optimal solutions",
    )


# =============================================================================
# Autouse fixture: suppress OR-Tools solver log output during tests
# =============================================================================


@pytest.fixture(autouse=True)
def _suppress_ortools_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress OR-Tools solver logs in tests.

    Handbook 11.2: CI must be deterministic. Solver logs contain
    nondeterministic wall-time output that varies across machines.
    """

    monkeypatch.setenv("ORTools_LogToStdout", "")
    monkeypatch.setenv("ORTools_LogToStderr", "")
