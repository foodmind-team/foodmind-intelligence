"""LLM integration package — provider-neutral local LLM adapters.

Modules:
  client     — OpenAI-compatible chat completions client (httpx)
  extractor  — LLMRecipeExtractor (RecipeExtractor Protocol)
  researcher — LLMKnowledgeResearcher (RecipeResearcher Protocol)
  explainer  — LLMPlanExplainer (schedule prose, additive)
"""

from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.llm.controller import LLMReActController
from cooking_plan_agent.llm.explainer import LLMPlanExplainer
from cooking_plan_agent.llm.extractor import LLMRecipeExtractor
from cooking_plan_agent.llm.researcher import LLMKnowledgeResearcher

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMPlanExplainer",
    "LLMReActController",
    "LLMRecipeExtractor",
    "LLMKnowledgeResearcher",
]
