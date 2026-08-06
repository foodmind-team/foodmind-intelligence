"""P5-1: ToolSpec / ToolRegistry 基座。"""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cooking_plan_agent.tooling.registry import ToolRegistry
from cooking_plan_agent.tooling.schemas import ToolSpec


class _FakeExtractor:
    """满足 RecipeExtractor Protocol 的假抽取器（无需真实实现）。"""

    async def extract(self, source_text: str) -> object:
        return {"dish_name": "Tofu"}


def test_tool_spec_name_must_be_snake_case():
    with pytest.raises(ValidationError):
        ToolSpec(name="Parse_recipe", description="d", parameters={"type": "object"})


def test_registry_specs_non_empty_with_extractor():
    context = SimpleNamespace(recipe_extractor=_FakeExtractor())
    registry = ToolRegistry(context)  # type: ignore[arg-type]
    specs = registry.specs()
    assert len(specs) >= 1
    assert any(tool.name == "parse_recipe" for tool in specs)


def test_registry_get_hits_and_misses():
    context = SimpleNamespace(recipe_extractor=_FakeExtractor())
    registry = ToolRegistry(context)  # type: ignore[arg-type]
    assert registry.get("parse_recipe") is not None
    assert registry.get("nope") is None


def test_registered_tool_schema_is_serialisable():
    # LLM 只消费 schema（name/description/parameters），executor 是内部实现。
    context = SimpleNamespace(recipe_extractor=_FakeExtractor())
    registry = ToolRegistry(context)  # type: ignore[arg-type]
    tool = registry.get("parse_recipe")
    assert tool is not None
    dumped = tool.model_dump(exclude={"executor"})
    assert dumped["name"] == "parse_recipe"
    assert "source_text" in dumped["parameters"]["properties"]
