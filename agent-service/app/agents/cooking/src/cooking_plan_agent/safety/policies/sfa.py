# =============================================================================
# 新加坡食品局（SFA）策略包（safety/policies/sfa）
# -----------------------------------------------------------------------------
# 新加坡（区域 "SG"）的食品安全阈值策略包，阈值遵循 SFA 公开指引，
# 由测试夹具锁定（计划 P3-04）。
# =============================================================================

"""Singapore Food Agency policy pack — Singapore (region "SG").

新加坡食品局策略包 —— 新加坡（区域 "SG"）。

Thresholds follow SFA public guidance (locked by test fixtures, plan P3-04):

阈值遵循 SFA 公开指引（由测试夹具锁定，计划 P3-04）：

  - Temperature Danger Zone is 5°C–60°C; keep hot food above 60°C and cold
    food below 5°C (SFA "Proper Temperature for Hot Foods").
  - Cooked/ready-to-eat hot food must not stay below 60°C for more than 4
    hours aggregate (Environmental Public Health (Food Hygiene) Regulations
    reg 13A — mandatory for catering).
  - Cook poultry and minced (ground) meat to a core temperature above 75°C;
    SFA publishes no granular per-cut table, so whole cuts / fish / eggs are
    not flagged by the protein-temperature rule under this pack.
  - Reheat food to at least 75°C for at least 2 minutes; food can only be
    reheated once.
  - SFA does not mandate a post-cooking rest time.

  - 温度危险区为 5°C–60°C；热食应保持在 60°C 以上，冷食保持在 5°C 以下
    （SFA《热食的适宜温度》）。
  - 已煮熟 / 即食的热食在 60°C 以下累计不得超过 4 小时
    （《环境卫生（食品卫生）条例》第 13A 条 —— 餐饮服务强制要求）。
  - 禽肉与绞肉（碎肉）应烹至中心温度高于 75°C；
    SFA 未发布分部位的细粒度表，因此本策略包下蛋白质温度规则
    不标记整块肉 / 鱼 / 蛋。
  - 复热应至少达到 75°C 并保持至少 2 分钟；食物只能复热一次。
  - SFA 未强制要求烹饪后的静置时间。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from cooking_plan_agent.safety.policy import PolicySource, SafetyPolicy, SafetyThresholds

# Poultry and minced meat must reach above 75°C (commonly applied SFA standard).
# Other categories are absent because SFA guidance gives no per-cut numbers —
# ProteinSafetyTemperatureRule skips categories not documented by the pack.
# 禽肉与绞肉必须达到 75°C 以上（广泛采用的 SFA 标准）。
# 其他类别缺失是因为 SFA 指引未给出分部位数值 ——
# ProteinSafetyTemperatureRule 会跳过策略包未记录的类别。
SFA_SAFE_MINIMUM_TEMPERATURES_C: dict[str, Decimal] = {
    "chicken": Decimal(75),
    "turkey": Decimal(75),
    "duck": Decimal(75),
    "goose": Decimal(75),
    "poultry": Decimal(75),
    "ground_beef": Decimal(75),
    "ground_pork": Decimal(75),
    "ground_lamb": Decimal(75),
    "ground_meat": Decimal(75),
}

SFA_POLICY = SafetyPolicy(
    region="SG",
    version="1.0",
    effective_at=date(2024, 1, 1),
    sources=(
        PolicySource(
            source_id="sfa-proper-temperature-hot-foods",
            title="Singapore Food Agency: Proper Temperature for Hot Foods",
            url="https://www.sfa.gov.sg/docs/default-source/educational-materials/proper-temperature-for-hot-foods.pdf",
        ),
        PolicySource(
            source_id="sfa-eph-food-hygiene-regs-13a",
            title="Environmental Public Health (Food Hygiene) Regulations, reg 13A (catered food)",
            url="https://sso.agc.gov.sg/SL/EPHA1987-RG16",
        ),
        PolicySource(
            source_id="sfa-catered-meals-guidelines",
            title="SFA: Guidelines for Ordering Catered Meals for Functions and Events",
            url="https://www.sfa.gov.sg/docs/default-source/our-services/guidelines-for-ordering-catered-meals-for-functions-and-events_070319.pdf",
        ),
    ),
    thresholds=SafetyThresholds(
        safe_minimum_temperatures_c=dict(SFA_SAFE_MINIMUM_TEMPERATURES_C),
        max_room_temp_holding_minutes=240,  # 4-hour aggregate rule (reg 13A)
        # 4 小时累计规则（第 13A 条）
        hot_holding_minimum_c=Decimal(60),
        cold_holding_maximum_c=Decimal(5),
        reheat_minimum_c=Decimal(75),
        reheat_hold_seconds=120,  # ≥75°C for ≥2 minutes
        # ≥75°C 保持 ≥2 分钟
        rest_time_minutes={},
    ),
)
