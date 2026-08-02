"""USDA FSIS policy pack — United States (region "US").

Thresholds mirror the USDA FSIS guidance that was previously hard-coded in
safety/rules.py; they are now versioned and traceable to official sources.

Internal temperatures (safe minimum internal cooking temperatures, °C):
  - Poultry (whole, parts, ground): 74°C (165°F)
  - Ground meats (beef/pork/lamb): 71°C (160°F)
  - Beef, pork, lamb, veal (steaks/chops/roasts): 63°C (145°F) + 3 min rest
  - Fish & shellfish: 63°C (145°F)
  - Eggs: 71°C (160°F)

Holding / danger-zone:
  - Danger zone 4°C–60°C; perishable food may sit at room temperature at most
    2 hours before bacteria multiply into the danger zone.
  - Hot holding ≥ 60°C; cold holding ≤ 4°C.
  - Reheat leftovers to at least 74°C (165°F) — USDA specifies no hold time.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from cooking_plan_agent.safety.policy import PolicySource, SafetyPolicy, SafetyThresholds

# Canonical USDA table — also the backward-compatible default for rules that
# are constructed without an explicit policy (existing unit tests).
USDA_SAFE_MINIMUM_TEMPERATURES_C: dict[str, Decimal] = {
    # Poultry (whole, parts, ground)
    "chicken": Decimal(74),
    "turkey": Decimal(74),
    "duck": Decimal(74),
    "goose": Decimal(74),
    "poultry": Decimal(74),
    # Ground meats (except poultry)
    "ground_beef": Decimal(71),
    "ground_pork": Decimal(71),
    "ground_lamb": Decimal(71),
    "ground_meat": Decimal(71),
    # Beef, pork, lamb, veal (steaks, chops, roasts)
    "beef": Decimal(63),
    "pork": Decimal(63),
    "lamb": Decimal(63),
    "veal": Decimal(63),
    # Fish & shellfish
    "fish": Decimal(63),
    "salmon": Decimal(63),
    "shrimp": Decimal(63),
    "shellfish": Decimal(63),
    # Eggs
    "egg": Decimal(71),
}

USDA_POLICY = SafetyPolicy(
    region="US",
    version="1.0",
    effective_at=date(2024, 1, 1),
    sources=(
        PolicySource(
            source_id="usda-fsis-safe-minimum-temps",
            title="USDA FSIS Safe Minimum Internal Temperature Chart",
            url="https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart",
        ),
        PolicySource(
            source_id="usda-fsis-danger-zone",
            title="USDA FSIS The Temperature Danger Zone (40°F–140°F)",
            url="https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/danger-zone-40f-140f",
        ),
        PolicySource(
            source_id="usda-fsis-rest-time",
            title="USDA FSIS Rest Time for Meat, Poultry, and Fish",
            url="https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/rest-time",
        ),
    ),
    thresholds=SafetyThresholds(
        safe_minimum_temperatures_c=dict(USDA_SAFE_MINIMUM_TEMPERATURES_C),
        max_room_temp_holding_minutes=120,  # 2-hour rule
        hot_holding_minimum_c=Decimal(60),
        cold_holding_maximum_c=Decimal(4),
        reheat_minimum_c=Decimal(74),  # 165°F
        reheat_hold_seconds=0,
        rest_time_minutes={"whole_cuts": 3},
    ),
)
