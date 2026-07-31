"""Inventory and resource feasibility — pure domain functions (no I/O).

Handbook sections 5.10–5.16: inventory sufficiency, FEFO allocation,
resource compatibility, and reservation proposals. All functions operate
on immutable Pydantic models from domain/models.py — no database, no
network, no side effects.
"""

from cooking_plan_agent.inventory.feasibility import (
    allocate_fefo,
    available_lot_quantity,
    build_reservation_proposal,
    check_all_inventory,
    check_required_resources,
    find_compatible_resources,
    is_lot_usable,
    resource_is_compatible,
)

__all__ = [
    "allocate_fefo",
    "available_lot_quantity",
    "build_reservation_proposal",
    "check_all_inventory",
    "check_required_resources",
    "find_compatible_resources",
    "is_lot_usable",
    "resource_is_compatible",
]
