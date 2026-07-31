"""Bounded web research module (handbook chapter 10).

Provider-neutral web search for filling cooking heat/duration gaps.
All concrete providers live in research/providers/ — the rest of the
codebase depends only on the RecipeResearcher Protocol.
"""

from cooking_plan_agent.research.researcher import Researcher

__all__ = ["Researcher"]
