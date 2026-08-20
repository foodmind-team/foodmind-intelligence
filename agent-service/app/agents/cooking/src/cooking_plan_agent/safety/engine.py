# =============================================================================
# 安全规则引擎（safety/engine）
# -----------------------------------------------------------------------------
# SafetyEngine 是安全验证的入口：它组合所有已注册的独立安全规则，
# 汇总各规则产生的 SafetyFinding，并归类整体安全状态，
# 供 LangGraph 路由层消费的 SafetyReport。
# =============================================================================

"""SafetyEngine — composes rules and produces a unified SafetyReport.

SafetyEngine —— 组合规则并产出统一的 SafetyReport。

Handbook 5.7–5.9: the engine is the entry point for safety validation.
It evaluates all registered rules independently, aggregates findings,
and classifies the overall safety status. Rules that produce
hard_repairable findings contribute required_safety_task_ids that the
merge_preparation node uses to inject sanitisation/temperature tasks.

手册 5.7–5.9：该引擎是安全验证的入口。
它独立评估所有已注册规则、汇总发现项，并归类整体安全状态。
产生 hard_repairable 发现的规则会贡献 required_safety_task_ids，
供 merge_preparation 节点用于注入消毒 / 温度任务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from cooking_plan_agent.domain.models import (
    SafetyContext,
    SafetyFinding,
    SafetyInsertion,
    SafetyReport,
)
from cooking_plan_agent.safety.policy import SafetyPolicy
from cooking_plan_agent.safety.rules import SafetyRule, default_rules


@dataclass(frozen=True)
class SafetyEngine:
    """Composable safety evaluation engine.

    可组合的安全评估引擎。

    Design:
      - Immutable after construction (frozen=True) — safe to share across nodes.
      - Rules are evaluated in registration order; findings preserve that order.
      - The engine itself has no state or side effects.
      - required_safety_task_ids are derived from hard_repairable findings:
        each finding that is repairable generates one task ID.

    设计：
      - 构造后不可变（frozen=True）—— 可在节点间安全共享。
      - 规则按注册顺序评估；发现项保持该顺序。
      - 引擎本身无状态、无副作用。
      - required_safety_task_ids 由 hard_repairable 发现派生：
        每个可修复的发现都会生成一个任务 ID。
    """

    rules: tuple[SafetyRule, ...] = field(default=default_rules)
    # P3-04: the regional policy pack whose thresholds the rules were built
    # from. When set, the produced SafetyReport records its region/version/
    # sources for traceability; None keeps legacy engine behaviour.
    # P3-04：规则所依据阈值的区域策略包。设置后，产出的 SafetyReport 会
    # 记录其 region/version/sources 以便追溯；None 则保持旧版引擎行为。
    policy: SafetyPolicy | None = None

    def evaluate(self, context: SafetyContext) -> SafetyReport:
        """Run all rules and produce an aggregated SafetyReport.

        运行所有规则并产出聚合的 SafetyReport。

        Args:
            context: Input context containing recipes, user allergens,
                     dietary restrictions, inventory, and cooking date.
            context：输入上下文，包含菜谱、用户过敏原、膳食限制、库存与烹饪日期。

        Returns:
            SafetyReport with all findings, overall safety status, and
            required safety task IDs for downstream injection.
            SafetyReport：包含全部发现项、整体安全状态，以及供下游注入的
            必需安全任务 ID。

        """
        findings: list[SafetyFinding] = []
        safety_task_ids: list[str] = []
        insertions: list[SafetyInsertion] = []

        for rule in self.rules:
            finding = rule.evaluate(context)
            if finding is not None:
                findings.append(finding)

                # Hard-repairable findings generate safety task IDs
                # for downstream nodes (merge_preparation) to inject as
                # actual CookingTask instances with safety_tags
                # hard_repairable 发现会生成安全任务 ID，
                # 供下游节点（merge_preparation）将其作为带 safety_tags 的
                # 实际 CookingTask 实例注入。
                if finding.severity == "hard_repairable":
                    task_id = f"safety_{finding.rule_id.lower()}_{uuid4().hex[:8]}"
                    safety_task_ids.append(task_id)
                    # P0-07: carry the structured insertion (with anchors)
                    # so merge_preparation can build the dependency chain.
                    # P0-07：携带结构化插入（含锚点），
                    # 以便 merge_preparation 构建依赖链。
                    if finding.insertion is not None:
                        insertions.append(finding.insertion)

        has_unrepairable = any(f.severity == "hard_unrepairable" for f in findings)
        # A plan is safe if no hard-level findings exist (repairable or not).
        # Warnings do not block — they surface but don't affect is_safe.
        # 若不存在任何 hard 级发现（无论是否可修复），计划即安全。
        # 警告不阻止 —— 它们会展示出来，但不影响 is_safe。
        is_safe = not any(f.severity.startswith("hard_") for f in findings)

        return SafetyReport(
            report_id=f"safety_{uuid4().hex[:12]}",
            findings=tuple(findings),
            is_safe=is_safe,
            has_unrepairable=has_unrepairable,
            required_safety_task_ids=tuple(safety_task_ids),
            insertions=tuple(insertions),
            safety_policy=self.policy.to_record() if self.policy is not None else None,
        )
