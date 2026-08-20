# =============================================================================
# 交叉污染规则（safety/cross_contamination）
# -----------------------------------------------------------------------------
# CrossContaminationRule：检测生蛋白质处理与即食食材处理在同一菜谱中共存，
# 并在两者之间注入消毒任务，阻断生食 → 即食的交叉污染风险。
# =============================================================================

"""Independently evaluable food-safety rule. 可独立评估的食品安全规则。"""

from __future__ import annotations

from dataclasses import dataclass

from cooking_plan_agent.domain.models import (
    SafetyContext,
    SafetyFinding,
    SafetyInsertion,
)
from cooking_plan_agent.safety.rule_support import (
    _SANITISE_DURATION_MINUTES,
    _SANITISE_REQUIRED_RESOURCES,
    _matches_keywords,
    _raw_protein_steps,
    _rte_steps,
)


@dataclass(frozen=True)
class CrossContaminationRule:
    """Detect raw protein handling near ready-to-eat ingredients.

    检测生蛋白质处理与即食食材相邻的情况。

    When a recipe uses raw proteins (meat, poultry, seafood, eggs) AND
    contains steps handling ready-to-eat items, a sanitisation task must
    be injected between them. This rule flags the violation; the
    merge_preparation node injects the sanitisation task.

    当菜谱使用生蛋白质（肉、禽、海鲜、蛋）且包含处理即食食材的步骤时，
    必须在两者之间注入消毒任务。本规则标记该违规；
    merge_preparation 节点负责注入消毒任务。

    Severity: hard_repairable — can always insert a sanitise step.

    严重级别：hard_repairable —— 总是可以插入消毒步骤。
    """

    rule_id: str = "SAFETY_CROSS_CONTAMINATION"

    # Ingredients considered "raw protein" — matched against canonical_name
    # 被视为“生蛋白质”的食材 —— 与 canonical_name 匹配
    _raw_protein_keywords: tuple[str, ...] = (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "veal",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "squid",
        "octopus",
        "egg",
        "meat",
        "poultry",
        "seafood",
    )

    # Step categories that imply ready-to-eat handling
    # 表示即食处理的步骤类别
    _rte_categories: tuple[str, ...] = (
        "plating",
        "garnishing",
        "dressing",
        "mixing",
    )

    def evaluate(self, context: SafetyContext) -> SafetyFinding | None:
        """Check each recipe for raw-protein / RTE co-existence.

        检查每个菜谱中是否存在生蛋白质 / 即食共存的情况。

        When both exist in the SAME recipe, locate the anchor steps:
          - after_step_number: last step that handles raw protein
          - before_step_number: first step that is RTE/plating
        The finding carries a structured SafetyInsertion so merge_preparation
        can build the raw → sanitise → RTE dependency chain (P0-07).

        当两者存在于同一菜谱时，定位锚点步骤：
          - after_step_number：最后一个处理生蛋白质的步骤
          - before_step_number：第一个即食 / 装盘步骤
        发现项携带结构化 SafetyInsertion，以便 merge_preparation
        构建 生食 → 消毒 → 即食 的依赖链（P0-07）。
        """
        for recipe in context.recipes:
            raw_steps = _raw_protein_steps(recipe, self._raw_protein_keywords)
            rte_steps = _rte_steps(recipe, self._rte_categories)

            if not raw_steps or not rte_steps:
                continue

            # Anchors: last raw step → sanitise → first RTE step.
            # 锚点：最后一个生食步骤 → 消毒 → 第一个即食步骤。
            after_step = raw_steps[-1].step_number
            before_step = rte_steps[0].step_number
            if after_step >= before_step:
                # Raw handling already precedes RTE within the same recipe
                # with no interleaving — still insert between them.
                # 同一菜谱中生食处理已先于即食且无交错 —— 仍在两者之间插入。
                pass

            raw_ingredients = [
                ing.raw_name
                for ing in recipe.ingredients
                if _matches_keywords(ing.canonical_name.lower(), self._raw_protein_keywords)
            ]

            insertion = SafetyInsertion(
                insertion_id=f"{self.rule_id.lower()}_{recipe.recipe_id}",
                rule_id=self.rule_id,
                recipe_id=recipe.recipe_id,
                after_step_number=after_step,
                before_step_number=before_step,
                task_instruction=(
                    "Sanitise cutting board and utensils after raw protein handling and before ready-to-eat assembly."
                ),
                duration_minutes=_SANITISE_DURATION_MINUTES,
                required_resources=_SANITISE_REQUIRED_RESOURCES,
            )

            return SafetyFinding(
                rule_id=self.rule_id,
                severity="hard_repairable",
                description=(
                    f"Cross-contamination risk: raw protein and ready-to-eat "
                    f"handling coexist in dish '{recipe.dish_name}' (steps "
                    f"{after_step} → {before_step}). A sanitisation task must "
                    f"be inserted between raw and RTE steps."
                ),
                affected_ingredient_names=tuple(raw_ingredients),
                recommended_action=(
                    "Insert a 'Sanitise cutting board and utensils' task between "
                    "raw protein handling and ready-to-eat assembly."
                ),
                insertion=insertion,
            )

        return None


# =============================================================================
# Rule 2: AllergenDetectionRule
# 规则 2：过敏原检测规则
# =============================================================================
