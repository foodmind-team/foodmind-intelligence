"""SafetyEngine — composes rules and produces a unified SafetyReport.

Handbook 5.7–5.9: the engine is the entry point for safety validation.
It evaluates all registered rules independently, aggregates findings,
and classifies the overall safety status. Rules that produce
hard_repairable findings contribute required_safety_task_ids that the
merge_preparation node uses to inject sanitisation/temperature tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from cooking_plan_agent.domain.models import SafetyContext, SafetyFinding, SafetyReport
from cooking_plan_agent.safety.rules import SafetyRule, default_rules


@dataclass(frozen=True)
class SafetyEngine:
    """Composable safety evaluation engine.

    Design:
      - Immutable after construction (frozen=True) — safe to share across nodes.
      - Rules are evaluated in registration order; findings preserve that order.
      - The engine itself has no state or side effects.
      - required_safety_task_ids are derived from hard_repairable findings:
        each finding that is repairable generates one task ID.
    """

    rules: tuple[SafetyRule, ...] = field(default=default_rules)

    def evaluate(self, context: SafetyContext) -> SafetyReport:
        """Run all rules and produce an aggregated SafetyReport.

        Args:
            context: Input context containing recipes, user allergens,
                     dietary restrictions, inventory, and cooking date.

        Returns:
            SafetyReport with all findings, overall safety status, and
            required safety task IDs for downstream injection.

        """
        findings: list[SafetyFinding] = []
        safety_task_ids: list[str] = []

        for rule in self.rules:
            finding = rule.evaluate(context)
            if finding is not None:
                findings.append(finding)

                # Hard-repairable findings generate safety task IDs
                # for downstream nodes (merge_preparation) to inject as
                # actual CookingTask instances with safety_tags
                if finding.severity == "hard_repairable":
                    task_id = f"safety_{finding.rule_id.lower()}_{uuid4().hex[:8]}"
                    safety_task_ids.append(task_id)

        has_unrepairable = any(f.severity == "hard_unrepairable" for f in findings)
        # A plan is safe if no hard-level findings exist (repairable or not).
        # Warnings do not block — they surface but don't affect is_safe.
        is_safe = not any(f.severity.startswith("hard_") for f in findings)

        return SafetyReport(
            report_id=f"safety_{uuid4().hex[:12]}",
            findings=tuple(findings),
            is_safe=is_safe,
            has_unrepairable=has_unrepairable,
            required_safety_task_ids=tuple(safety_task_ids),
        )
