"""Small value objects used by repair services."""

from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.models import StrictModel


class Shortage(NamedTuple):
    """A single resource or ingredient shortage."""

    item: str
    """Ingredient name or resource type."""
    required: Decimal
    """Amount needed."""
    available: Decimal
    """Amount available."""
    unit: str
    """Unit of measure."""


class RepairValidation(StrictModel):
    """Result of validating a single RepairOption."""

    is_valid: bool
    """Whether the option is internally consistent."""
    issues: tuple[str, ...] = ()
    """Validation issues, if any."""


# =============================================================================
# 5.17  Calculate exact shortages
# =============================================================================
