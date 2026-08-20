# =============================================================================
# 蛋白质安全温度规则（safety/temperatures）
# -----------------------------------------------------------------------------
# ProteinSafetyTemperatureRule：验证蛋白质烹饪步骤是否达到所在区域的
# 安全最低内部温度。未指定温度或低于阈值的步骤会被标记，并给出建议温度。
# =============================================================================

"""Independently evaluable food-safety rule. 可独立评估的食品安全规则。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from cooking_plan_agent.domain.models import (
    SafetyContext,
    SafetyFinding,
)
from cooking_plan_agent.safety.policies.usda import USDA_SAFE_MINIMUM_TEMPERATURES_C
from cooking_plan_agent.safety.rule_support import _dominant_protein_type, _is_protein_heating_step


@dataclass(frozen=True)
class ProteinSafetyTemperatureRule:
    """Verify that protein cooking steps reach the region's safe internal temperatures.

    验证蛋白质烹饪步骤是否达到所在区域的安全最低内部温度。

    For each recipe step that involves heating a protein, check that
    target_temperature_c is at or above the safe minimum defined by the
    active regional policy (P3-04). Steps without a specified temperature are
    flagged with the recommended safe temp. Protein categories the policy does
    not document are skipped (never flagged).

    对每个涉及加热蛋白质的菜谱步骤，检查 target_temperature_c 是否达到或超过
    当前区域策略（P3-04）定义的安全最低值。未指定温度的步骤会被标记并给出
    建议安全温度。策略未记录的蛋白质类别会被跳过（从不标记）。

    Severity: hard_repairable — temperature can always be specified.

    严重级别：hard_repairable —— 温度总是可以指定。
    """

    rule_id: str = "SAFETY_PROTEIN_TEMPERATURE"

    # Per-protein safe minimum internal temperatures (°C). Backward-compatible
    # default is the USDA pack; production binds the resolved regional policy.
    # 每种蛋白质的安全最低内部温度（°C）。向后兼容的默认值为 USDA 包；
    # 生产环境绑定已解析的区域策略。
    safe_temperatures_c: dict[str, Decimal] = field(default_factory=lambda: dict(USDA_SAFE_MINIMUM_TEMPERATURES_C))

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check all protein heating steps across recipes. 检查所有菜谱中的蛋白质加热步骤。"""
        unsafe_steps: list[str] = []

        for recipe in context.recipes:
            for step in recipe.steps:
                if not _is_protein_heating_step(step):
                    continue

                # Determine protein type from the recipe ingredients
                # 从菜谱食材中确定蛋白质类型
                protein_type = _dominant_protein_type(recipe)
                safe_temp = self.safe_temperatures_c.get(protein_type)

                if safe_temp is None:
                    continue  # Not a tracked protein — skip
                    # 不是被追踪的蛋白质 —— 跳过

                if step.target_temperature_c is None:
                    unsafe_steps.append(
                        f"'{recipe.dish_name}' step {step.step_number}: "
                        f"no target temperature specified — "
                        f"recommend ≥{safe_temp}°C for {protein_type}"
                    )
                elif step.target_temperature_c < safe_temp:
                    unsafe_steps.append(
                        f"'{recipe.dish_name}' step {step.step_number}: "
                        f"target {step.target_temperature_c}°C is below "
                        f"safe minimum {safe_temp}°C for {protein_type}"
                    )

        if not unsafe_steps:
            return None

        detail = "; ".join(unsafe_steps)
        return SafetyFinding(
            rule_id=self.rule_id,
            severity="hard_repairable",
            description=(f"Protein cooking temperature below regional safe minimum: {detail}"),
            recommended_action=(
                "Set target temperatures to at or above the safe minimum "
                "internal temperatures defined by the active regional safety "
                "policy (see the plan's policy sources)."
            ),
        )


# =============================================================================
# Rule 4: DietaryCompatibilityRule
# 规则 4：膳食兼容性规则
# =============================================================================
