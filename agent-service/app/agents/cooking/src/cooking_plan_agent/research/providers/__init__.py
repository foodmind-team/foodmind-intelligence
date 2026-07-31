"""Concrete search providers for web research.

Each provider implements the SearchProvider Protocol defined in researcher.py.
They normalise provider-specific results into SearchDocument.
"""

from cooking_plan_agent.research.providers.fake import FakeSearchProvider

__all__ = ["FakeSearchProvider"]
