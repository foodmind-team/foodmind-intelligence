"""Quick script to show parsed output of golden fixtures."""
import asyncio
from cooking_plan_agent.parsing.extractor import RecipeExtractor
from cooking_plan_agent.parsing.golden_fixtures import GOLDEN_FIXTURES


async def main() -> None:
    extractor = RecipeExtractor()
    for name in ["crab_legs", "garlic_shrimp", "rib_soup"]:
        text = GOLDEN_FIXTURES[name]
        c = await extractor.extract(text)
        print(f"===== {name}: {c.dish_name} ({c.source_language}, servings={c.original_servings}) =====")
        print(f"Source: {c.extraction_source}")
        print(f"Ingredients ({len(c.ingredients)}):")
        for ing in c.ingredients:
            qty = f"{ing.quantity}{ing.unit}" if ing.quantity else "?"
            prep = f", {ing.preparation}" if ing.preparation else ""
            print(f"  {qty} {ing.name}{prep}  [conf={ing.confidence}]")
        print(f"Steps ({len(c.steps)}):")
        for s in c.steps:
            parts = [s.category]
            if s.heat_level.value != "NONE":
                parts.append(f"heat={s.heat_level.value}")
            if s.active_duration_minutes:
                parts.append(f"active={s.active_duration_minutes}m")
            if s.passive_duration_minutes:
                parts.append(f"passive={s.passive_duration_minutes}m")
            if s.target_temperature_c:
                parts.append(f"temp={s.target_temperature_c}C")
            if s.resources_hint:
                parts.append(f"tools={','.join(s.resources_hint)}")
            print(f"  Step{s.step_number} [{', '.join(parts)}]")
            print(f"    {s.instruction[:100]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
