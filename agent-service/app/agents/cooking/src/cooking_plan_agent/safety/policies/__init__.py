# =============================================================================
# 已注册的区域食品安全策略包（safety/policies/__init__）
# -----------------------------------------------------------------------------
# 导入本包时会在导入期注册所有已批准的策略包，使 safety/policy.resolve_policy
# 能够提供这些策略。新增区域 / 版本时，只需在此添加模块并在下方导入即可，
# 无需其他接线。
# =============================================================================

"""Registered regional safety policy packs (P3-04).

已注册的区域食品安全策略包（P3-04）。

Importing this package registers every approved pack at import time, so
``safety/policy.resolve_policy`` can serve them. Adding a new region/version
means adding a module here and importing it below — no other wiring.

导入本包会在导入期注册每一个已批准的策略包，使 ``safety/policy.resolve_policy``
能够提供这些策略。新增区域 / 版本意味着在此添加一个模块并在下方导入它 ——
无需其他接线。
"""

from cooking_plan_agent.safety.policies.sfa import SFA_POLICY
from cooking_plan_agent.safety.policies.usda import USDA_POLICY
from cooking_plan_agent.safety.policy import register_policy

register_policy(USDA_POLICY)
register_policy(SFA_POLICY)

__all__ = ["SFA_POLICY", "USDA_POLICY"]
