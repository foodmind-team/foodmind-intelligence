"""Registered regional safety policy packs (P3-04).

Importing this package registers every approved pack at import time, so
``safety/policy.resolve_policy`` can serve them. Adding a new region/version
means adding a module here and importing it below — no other wiring.
"""

from cooking_plan_agent.safety.policies.sfa import SFA_POLICY
from cooking_plan_agent.safety.policies.usda import USDA_POLICY
from cooking_plan_agent.safety.policy import register_policy

register_policy(USDA_POLICY)
register_policy(SFA_POLICY)

__all__ = ["SFA_POLICY", "USDA_POLICY"]
