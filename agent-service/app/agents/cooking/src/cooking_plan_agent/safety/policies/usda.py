# =============================================================================
# 美国农业部（USDA FSIS）策略包（safety/policies/usda）
# -----------------------------------------------------------------------------
# 美国（区域 "US"）的食品安全阈值策略包。阈值沿用此前硬编码在
# safety/rules.py 中的 USDA FSIS 指引，现已带版本并可追溯至官方来源。
# =============================================================================

"""USDA FSIS policy pack — United States (region "US").

美国农业部食品安全检验局（USDA FSIS）策略包 —— 美国（区域 "US"）。

Thresholds mirror the USDA FSIS guidance that was previously hard-coded in
safety/rules.py; they are now versioned and traceable to official sources.

阈值沿用此前硬编码在 safety/rules.py 中的 USDA FSIS 指引；
现已带版本并可追溯至官方来源。

Internal temperatures (safe minimum internal cooking temperatures, °C):
  - Poultry (whole, parts, ground): 74°C (165°F)
  - Ground meats (beef/pork/lamb): 71°C (160°F)
  - Beef, pork, lamb, veal (steaks/chops/roasts): 63°C (145°F) + 3 min rest
  - Fish & shellfish: 63°C (145°F)
  - Eggs: 71°C (160°F)

内部温度（安全最低内部烹饪温度，°C）：
  - 禽肉（整只、部位、绞肉）：74°C（165°F）
  - 绞肉（牛肉 / 猪肉 / 羊肉）：71°C（160°F）
  - 牛、猪、羊、犊牛肉（牛排 / 排骨 / 烤肉）：63°C（145°F）+ 3 分钟静置
  - 鱼与贝类：63°C（145°F）
  - 蛋：71°C（160°F）

Holding / danger-zone:
  - Danger zone 4°C–60°C; perishable food may sit at room temperature at most
    2 hours before bacteria multiply into the danger zone.
  - Hot holding ≥ 60°C; cold holding ≤ 4°C.
  - Reheat leftovers to at least 74°C (165°F) — USDA specifies no hold time.

保温 / 危险温度区：
  - 危险区为 4°C–60°C；易腐食品在室温下最多放置 2 小时，
    之后细菌会繁殖进入危险区。
  - 热保温 ≥ 60°C；冷保温 ≤ 4°C。
  - 剩菜复热至至少 74°C（165°F）—— USDA 未规定保持时间。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from cooking_plan_agent.safety.policy import PolicySource, SafetyPolicy, SafetyThresholds

# Canonical USDA table — also the backward-compatible default for rules that
# are constructed without an explicit policy (existing unit tests).
# 规范 USDA 表 —— 也是未显式指定策略构造的规则的向后兼容默认值（现有单元测试）。
USDA_SAFE_MINIMUM_TEMPERATURES_C: dict[str, Decimal] = {
    # Poultry (whole, parts, ground)
    # 禽肉（整只、部位、绞肉）
    "chicken": Decimal(74),
    "turkey": Decimal(74),
    "duck": Decimal(74),
    "goose": Decimal(74),
    "poultry": Decimal(74),
    # Ground meats (except poultry)
    # 绞肉（禽肉除外）
    "ground_beef": Decimal(71),
    "ground_pork": Decimal(71),
    "ground_lamb": Decimal(71),
    "ground_meat": Decimal(71),
    # Beef, pork, lamb, veal (steaks, chops, roasts)
    # 牛、猪、羊、犊牛肉（牛排、排骨、烤肉）
    "beef": Decimal(63),
    "pork": Decimal(63),
    "lamb": Decimal(63),
    "veal": Decimal(63),
    # Fish & shellfish
    # 鱼与贝类
    "fish": Decimal(63),
    "salmon": Decimal(63),
    "shrimp": Decimal(63),
    "shellfish": Decimal(63),
    # Eggs
    # 蛋
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
        # 两小时规则
        hot_holding_minimum_c=Decimal(60),
        cold_holding_maximum_c=Decimal(4),
        reheat_minimum_c=Decimal(74),  # 165°F
        # 165°F
        reheat_hold_seconds=0,
        rest_time_minutes={"whole_cuts": 3},
    ),
)
