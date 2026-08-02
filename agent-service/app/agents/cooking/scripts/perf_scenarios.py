"""Performance scenario generator for scheduling benchmark and stress tests.

Generates realistic multi-recipe, multi-serving workloads with configurable:
  - Number of recipes (1–20)
  - Number of steps per recipe (2–20)
  - Number of ingredients per recipe (3–15)
  - Target servings (1–8)
  - Ingredient catalogue size
  - Step patterns (boil, bake, marinate, stir_fry, simmer, simple)

Output: a GeneratePlanRequest payload ready for the /generate endpoint,
and a SchedulingProblem for direct CP-SAT benchmarking.

Usage:
    # Generate a 3-dish family dinner scenario
    python scripts/perf_scenarios.py --recipes 3 --steps 6 --servings 4

    # Generate a stress test: 12 dishes, large catalogue
    python scripts/perf_scenarios.py --recipes 12 --steps 8 --catalogue 80

    # Export as JSON for API testing
    python scripts/perf_scenarios.py --recipes 3 --output scenario.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    ResourceNeed,
    TaskDependency,
)
from cooking_plan_agent.scheduling.models import SchedulingProblem

# =============================================================================
# Scenario configuration — tweak these to match target workloads
# =============================================================================

# Default kitchen: 4-burner stove, 1 oven, 1 sink, 1 microwave
DEFAULT_KITCHEN: tuple[KitchenResourceSnapshot, ...] = (
    KitchenResourceSnapshot(
        resource_id="stove:main",
        resource_type="stove",
        capacity=Decimal(4),
        capacity_unit="burners",
        capabilities=("gas",),
    ),
    KitchenResourceSnapshot(
        resource_id="oven:main",
        resource_type="oven",
        capacity=Decimal(1),
        capabilities=("convection",),
    ),
    KitchenResourceSnapshot(
        resource_id="sink:main",
        resource_type="sink",
        capacity=Decimal(2),
        capacity_unit="basins",
    ),
    KitchenResourceSnapshot(
        resource_id="microwave:main",
        resource_type="microwave",
        capacity=Decimal(1),
    ),
)

# Ingredient catalogue — realistic names for scenario generation
_INGREDIENTS = [
    ("chicken breast", "g"),
    ("chicken thigh", "g"),
    ("beef sirloin", "g"),
    ("pork belly", "g"),
    ("salmon fillet", "g"),
    ("shrimp", "g"),
    ("egg", "piece"),
    ("tofu", "g"),
    ("rice", "g"),
    ("pasta", "g"),
    ("flour", "g"),
    ("onion", "piece"),
    ("garlic", "piece"),
    ("ginger", "piece"),
    ("tomato", "piece"),
    ("bell pepper", "piece"),
    ("carrot", "piece"),
    ("broccoli", "g"),
    ("cabbage", "g"),
    ("spinach", "g"),
    ("mushroom", "g"),
    ("potato", "piece"),
    ("soy sauce", "ml"),
    ("oyster sauce", "ml"),
    ("sesame oil", "ml"),
    ("olive oil", "ml"),
    ("vinegar", "ml"),
    ("salt", "g"),
    ("sugar", "g"),
    ("black pepper", "g"),
    ("chilli", "piece"),
    ("butter", "g"),
    ("milk", "ml"),
    ("cream", "ml"),
    ("cheese", "g"),
    ("lemon", "piece"),
    ("lime", "piece"),
    ("cilantro", "g"),
    ("basil", "g"),
    ("cornstarch", "g"),
]

# Dish name templates
_DISH_TEMPLATES = [
    "Stir-Fried {protein} with {veg}",
    "Braised {protein} in {sauce} Sauce",
    "{protein} and {veg} Soup",
    "Pan-Seared {protein} with {garnish}",
    "Oven-Roasted {protein} with {veg}",
    "{protein} {veg} Hot Pot",
    "Steamed {protein} with {veg}",
    "{veg} Salad with Grilled {protein}",
    "{protein} Curry with {veg}",
    "Crispy {protein} with {sauce} Glaze",
    "{protein} {veg} Stir-Fry",
    "Garlic {protein} with {veg}",
    "Spicy {protein} with {veg}",
    "Sweet and Sour {protein}",
    "{protein} Congee with {veg}",
]

_PROTEINS = ["Chicken", "Beef", "Pork", "Shrimp", "Tofu", "Fish"]
_VEGETABLES = ["Broccoli", "Cabbage", "Spinach", "Mushroom", "Bell Pepper", "Carrot", "Tomato", "Onion"]
_SAUCES = ["Soy", "Oyster", "Garlic", "Chilli", "Black Bean", "Hoisin"]
_GARNISHES = ["Cilantro", "Spring Onion", "Sesame Seeds", "Basil", "Chilli Flakes"]


# =============================================================================
# Public API
# =============================================================================


def generate_perf_scenario(
    recipes: int = 3,
    steps_per_recipe: int = 6,
    ingredients_per_recipe: int = 8,
    target_servings: int = 4,
    catalogue_size: int = 30,
    time_limit_minutes: int | None = None,
    seed: int | None = None,
) -> ScenarioResult:
    """Generate a realistic performance-test scenario.

    Args:
        recipes: Number of recipes (1–20).
        steps_per_recipe: Steps per recipe (2–20).
        ingredients_per_recipe: Ingredients per recipe (3–15).
        target_servings: Target serving count (1–8).
        catalogue_size: How many ingredients in the catalogue (1–40).
        time_limit_minutes: Optional hard deadline. Defaults to None (no limit).
        seed: Random seed for reproducibility.

    Returns:
        ScenarioResult with request, scheduling problem, and metadata.
    """
    if seed is not None:
        random.seed(seed)

    # Pick a subset of ingredients as the catalogue for this scenario
    catalogue = random.sample(_INGREDIENTS, min(catalogue_size, len(_INGREDIENTS)))

    generated_tasks: list[CookingTask] = []
    recipe_texts: list[dict] = []
    inventory_lots: list[InventoryLotSnapshot] = []

    for r_idx in range(recipes):
        # Dish name
        protein = random.choice(_PROTEINS)
        veg = random.choice(_VEGETABLES)
        sauce = random.choice(_SAUCES)
        garnish = random.choice(_GARNISHES)
        dish_name = random.choice(_DISH_TEMPLATES).format(
            protein=protein,
            veg=veg,
            sauce=sauce,
            garnish=garnish,
        )
        recipe_id = f"perf_recipe_{r_idx + 1}"

        # Ingredients for this recipe
        recipe_ingredients = random.sample(catalogue, min(ingredients_per_recipe, len(catalogue)))
        ingredient_lines: list[str] = []
        for ing_name, ing_unit in recipe_ingredients:
            qty = random.randint(50, 500)
            ingredient_lines.append(f"{qty}{ing_unit} {ing_name}")
            # Create inventory lot with generous stock
            inventory_lots.append(
                InventoryLotSnapshot(
                    lot_id=f"lot_{ing_name.replace(' ', '_')}_{r_idx}",
                    item_id=f"item_{ing_name.replace(' ', '_')}",
                    canonical_name=ing_name,
                    on_hand=Decimal(random.randint(500, 2000)),
                    reserved=Decimal(0),
                    unit=ing_unit,
                    expiry_date=datetime.now(UTC).date() + timedelta(days=random.randint(5, 60)),
                )
            )

        # Steps for this recipe
        step_patterns = random.choices(
            ["simple", "boil", "stir_fry", "simmer", "bake", "marinate"],
            k=steps_per_recipe,
        )
        step_lines: list[str] = []
        for s_idx, pattern in enumerate(step_patterns):
            task_id = f"{recipe_id}_s{s_idx + 1}"
            step_text, task = _generate_step(
                task_id,
                recipe_id,
                pattern,
                s_idx + 1,
                steps_per_recipe,
                s_idx > 0,
                task_id if s_idx == 0 else generated_tasks[-1].task_id,
            )
            step_lines.append(f"{s_idx + 1}. {step_text}")
            if task is not None:
                generated_tasks.append(task)

        # Build recipe text (preprocessed format)
        text = f"{dish_name}\n\n食材：\n" + "\n".join(ingredient_lines) + "\n\n步骤：\n" + "\n".join(step_lines)
        recipe_texts.append(
            {
                "recipe_id": recipe_id,
                "text": text,
                "target_servings": target_servings,
            }
        )

    # Build SchedulingProblem
    problem = SchedulingProblem(
        tasks=tuple(generated_tasks),
        resources=DEFAULT_KITCHEN,
        requested_time_limit_minutes=time_limit_minutes,
        solver_timeout_seconds=5.0,
    )

    # Build approximate GeneratePlanRequest
    request_payload = {
        "request_id": f"perf_{recipes}d_{steps_per_recipe}s_{seed or 0}",
        "user_id": "perf_test_user",
        "recipes": tuple(recipe_texts),
        "dietary_restrictions": (),
        "user_allergens": (),
        "time_limit_minutes": time_limit_minutes,
        "inventory_lots": tuple(lot.model_dump(mode="json") for lot in inventory_lots),
        "kitchen_resources": tuple(r.model_dump(mode="json") for r in DEFAULT_KITCHEN),
        "schema_version": "1.0",
    }

    return ScenarioResult(
        request=request_payload,
        problem=problem,
        total_tasks=len(generated_tasks),
        total_recipes=recipes,
        seed=seed or 0,
    )


# =============================================================================
# ScenarioResult
# =============================================================================


class ScenarioResult:
    """Container for a generated performance scenario."""

    def __init__(
        self,
        request: dict,
        problem: SchedulingProblem,
        total_tasks: int,
        total_recipes: int,
        seed: int,
    ) -> None:
        self.request = request
        self.problem = problem
        self.total_tasks = total_tasks
        self.total_recipes = total_recipes
        self.seed = seed

    def summary(self) -> str:
        """One-line description for logging."""
        return f"Scenario(seed={self.seed}, recipes={self.total_recipes}, tasks={self.total_tasks})"


# =============================================================================
# Step generation helpers
# =============================================================================


def _generate_step(
    task_id: str,
    recipe_id: str,
    pattern: str,
    step_num: int,
    total_steps: int,
    has_prev: bool,
    prev_task_id: str,
) -> tuple[str, CookingTask | None]:
    """Generate a task and its human-readable instruction for one step pattern."""
    deps: tuple[TaskDependency, ...] = ()
    if has_prev and step_num > 1:
        deps = (TaskDependency(predecessor_id=prev_task_id),)

    if pattern == "boil":
        passive_dur = random.choice([8, 10, 12, 15, 20])
        text = f"Bring water to a boil and cook for {passive_dur} minutes over high heat."
        task = CookingTask(
            task_id=task_id,
            dish_id=recipe_id,
            instruction=text,
            duration_minutes=passive_dur,
            work_mode=WorkMode.PASSIVE,
            category="heating",
            heat_level=HeatLevel.HIGH,
            target_temperature_c=Decimal(100),
            dependencies=deps,
            resources=(ResourceNeed(quantity=1, resource_type="stove"),),
        )
        return text, task

    if pattern == "bake":
        passive_dur = random.choice([15, 20, 25, 30, 40])
        temp = Decimal(random.choice([160, 180, 200, 220]))
        text = f"Bake in preheated oven at {temp}°C for {passive_dur} minutes."
        task = CookingTask(
            task_id=task_id,
            dish_id=recipe_id,
            instruction=text,
            duration_minutes=passive_dur,
            work_mode=WorkMode.PASSIVE,
            category="heating",
            heat_level=HeatLevel.MEDIUM,
            target_temperature_c=temp,
            dependencies=deps,
            resources=(ResourceNeed(quantity=1, resource_type="oven"),),
        )
        return text, task

    if pattern == "stir_fry":
        active_dur = random.choice([3, 4, 5, 6, 8])
        text = f"Stir-fry over high heat for {active_dur} minutes until done."
        task = CookingTask(
            task_id=task_id,
            dish_id=recipe_id,
            instruction=text,
            duration_minutes=active_dur,
            work_mode=WorkMode.ACTIVE,
            category="heating",
            heat_level=HeatLevel.HIGH,
            dependencies=deps,
            resources=(
                ResourceNeed(quantity=1, resource_type="stove"),
                ResourceNeed(quantity=1, resource_type="pan"),
            ),
        )
        return text, task

    if pattern == "simmer":
        passive_dur = random.choice([20, 30, 40, 60, 90])
        text = f"Reduce heat and simmer for {passive_dur} minutes, stirring occasionally."
        task = CookingTask(
            task_id=task_id,
            dish_id=recipe_id,
            instruction=text,
            duration_minutes=passive_dur,
            work_mode=WorkMode.PASSIVE,
            category="heating",
            heat_level=HeatLevel.LOW,
            dependencies=deps,
            resources=(ResourceNeed(quantity=1, resource_type="stove"),),
        )
        return text, task

    if pattern == "marinate":
        passive_dur = random.choice([15, 20, 30, 60])
        text = f"Marinate and let rest for {passive_dur} minutes."
        task = CookingTask(
            task_id=task_id,
            dish_id=recipe_id,
            instruction=text,
            duration_minutes=passive_dur,
            work_mode=WorkMode.PASSIVE,
            category="resting",
            dependencies=deps,
        )
        return text, task

    # Default: simple active task (chopping, mixing, etc.)
    active_dur = random.choice([2, 3, 4, 5])
    verbs = ["Chop", "Dice", "Mix", "Slice", "Mince", "Grate", "Peel", "Wash"]
    verb = random.choice(verbs)
    text = f"{verb} ingredients (active, {active_dur} min)."
    task = CookingTask(
        task_id=task_id,
        dish_id=recipe_id,
        instruction=text,
        duration_minutes=active_dur,
        work_mode=WorkMode.ACTIVE,
        category="preparation",
        dependencies=deps,
        resources=(ResourceNeed(quantity=1, resource_type="cutting_board"),),
    )
    return text, task


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance test scenarios.")
    parser.add_argument("--recipes", type=int, default=3, help="Number of recipes (default: 3)")
    parser.add_argument("--steps", type=int, default=6, help="Steps per recipe (default: 6)")
    parser.add_argument("--ingredients", type=int, default=8, help="Ingredients per recipe (default: 8)")
    parser.add_argument("--servings", type=int, default=4, help="Target servings (default: 4)")
    parser.add_argument("--catalogue", type=int, default=30, help="Catalogue size (default: 30)")
    parser.add_argument("--time-limit", type=int, default=None, help="Hard time limit in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    scenario = generate_perf_scenario(
        recipes=args.recipes,
        steps_per_recipe=args.steps,
        ingredients_per_recipe=args.ingredients,
        target_servings=args.servings,
        catalogue_size=args.catalogue,
        time_limit_minutes=args.time_limit,
        seed=args.seed,
    )

    print("=== Performance Scenario ===")
    print(f"  Recipes:  {scenario.total_recipes}")
    print(f"  Tasks:    {scenario.total_tasks}")
    print(f"  Servings: {args.servings}")
    print(f"  Seed:     {scenario.seed}")
    print(f"  Time limit: {args.time_limit or 'none'}")

    if args.output:
        payload = {
            "request": scenario.request,
            "scheduling_problem": {
                "task_count": len(scenario.problem.tasks),
                "resource_count": len(scenario.problem.resources),
            },
            "meta": {
                "recipes": scenario.total_recipes,
                "tasks": scenario.total_tasks,
                "seed": scenario.seed,
            },
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nExported to: {args.output}")


if __name__ == "__main__":
    main()
