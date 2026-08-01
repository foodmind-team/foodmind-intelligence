"""Configuration module — environment-driven application settings.

Re-exports the public API so callers only need one import:
  >>> from cooking_plan_agent.config import Settings, get_settings
"""

from cooking_plan_agent.config.settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
