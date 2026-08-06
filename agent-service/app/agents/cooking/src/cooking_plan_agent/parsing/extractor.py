"""Rule-based recipe extraction adapter.

Parsing patterns live in ``extractor_patterns`` so this module focuses on
turning text into domain candidates.
"""

from __future__ import annotations

import re
from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import ExtractedIngredient, ExtractedRecipeCandidate, ExtractedStep
from cooking_plan_agent.parsing.extractor_patterns import (
    _CHINESE_UNIT_MAP,
    _HEAT_LEVELS,
    _INGREDIENT_HEADER_PATTERNS,
    _PREP_KEYWORDS,
    _RE_DURATION_RANGE,
    _RE_DURATION_SINGLE,
    _RE_INGREDIENT_CHINESE,
    _RE_INGREDIENT_WESTERN,
    _RE_NAME_PREP_SPLIT,
    _RE_NO_QUANTITY,
    _RE_PAREN_NOTE,
    _RE_QUANTITY_QUALIFIER,
    _RE_STEP_NUMBER,
    _RE_TRAILING_PUNCT,
    _RE_TRAILING_QUANTITY,
    _RESOURCE_KEYWORDS,
    _STEP_HEADER_PATTERNS,
    _TECHNIQUE_PATTERNS,
)


class RecipeExtractor:
    """Rule-based recipe text extractor implementing the RecipeExtractor Protocol.

    Extracts structured ExtractedRecipeCandidate from preprocessed recipe text.
    No LLM dependency — uses regex and keyword matching for MVP.

    Protocol contract (from workflow/context.py):
        async def extract(self, source_text: str) -> ExtractedRecipeCandidate
    """

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        """Parse preprocessed recipe text into a structured candidate.

        Args:
            source_text: Preprocessed recipe text (decoded, normalized, cleaned).

        Returns:
            ExtractedRecipeCandidate with ingredients and steps.
        """
        lines = source_text.strip().split("\n")

        # Separate ingredient and step sections
        ingredient_lines, step_lines = self._split_sections(lines)

        # Parse each section; expand "味精/鸡精" style slash alternatives
        # into separate ingredient candidates so inventory matching works.
        parsed_ingredients = tuple(self._parse_ingredient(line) for line in ingredient_lines)
        ingredients = tuple(
            sub for ing in parsed_ingredients if ing is not None for sub in self._expand_slash_alternatives(ing)
        )

        steps = tuple(self._parse_step(i + 1, line) for i, line in enumerate(step_lines))

        # Generate a stable recipe_id from the first line (dish name)
        dish_name = self._extract_dish_name(lines)
        recipe_id = self._make_recipe_id(dish_name)

        # Detect language from the text content
        source_language = self._detect_language(source_text)

        # Default to 2 servings when not specified
        original_servings = self._extract_servings(source_text)

        return ExtractedRecipeCandidate(
            recipe_id=recipe_id,
            dish_name=dish_name,
            original_servings=original_servings,
            source_language=source_language,
            ingredients=ingredients,
            steps=steps,
            extraction_source="RULE_BASED",
        )

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sections(lines: list[str]) -> tuple[list[str], list[str]]:
        """Split recipe lines into ingredient and step sections.

        Detects section headers ("Ingredients:", "Steps:", "食材:", "步骤:") to
        determine boundaries. Falls back to heuristics if no headers found.
        """
        ingredient_start: int | None = None
        step_start: int | None = None

        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            # Check ingredient headers
            if ingredient_start is None:
                for pat in _INGREDIENT_HEADER_PATTERNS:
                    if pat.match(stripped):
                        ingredient_start = i
                        break
            # Check step headers
            if step_start is None:
                for pat in _STEP_HEADER_PATTERNS:
                    if pat.match(stripped):
                        step_start = i
                        break

        if ingredient_start is not None and step_start is not None:
            # Both sections found
            ing_lines = lines[ingredient_start + 1 : step_start]
            step_lines = lines[step_start + 1 :]
        elif step_start is not None:
            # Only step section found — everything before is ingredients
            ing_lines = lines[:step_start]
            step_lines = lines[step_start + 1 :]
        else:
            # No explicit sections — use heuristics
            ing_lines, step_lines = RecipeExtractor._heuristic_split(lines)

        # Filter empty lines and strip
        ing_lines = [line.strip() for line in ing_lines if line.strip()]
        step_lines = [line.strip() for line in step_lines if line.strip()]

        return ing_lines, step_lines

    @staticmethod
    def _heuristic_split(lines: list[str]) -> tuple[list[str], list[str]]:
        """Heuristic split: numbered lines are steps, everything else is ingredients."""
        ing_lines: list[str] = []
        step_lines: list[str] = []

        found_first_step = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if _RE_STEP_NUMBER.match(stripped):
                found_first_step = True
                step_lines.append(stripped)
            elif found_first_step:
                step_lines.append(stripped)
            else:
                ing_lines.append(stripped)

        # If no numbered steps found, treat all as a single block
        if not step_lines:
            step_lines = ing_lines
            ing_lines = []

        return ing_lines, step_lines

    # ------------------------------------------------------------------
    # Ingredient parsing
    # ------------------------------------------------------------------

    def _parse_ingredient(self, line: str) -> ExtractedIngredient | None:
        """Parse a single ingredient line into ExtractedIngredient.

        Returns None if the line cannot be parsed as an ingredient.
        """
        if not line.strip():
            return None

        # Skip section headers
        stripped_lower = line.strip().lower()
        for pat in _INGREDIENT_HEADER_PATTERNS:
            if pat.match(stripped_lower):
                return None
        for pat in _STEP_HEADER_PATTERNS:
            if pat.match(stripped_lower):
                return None

        # Try Western pattern first, then Chinese
        result = self._try_western(line) or self._try_chinese(line) or self._try_no_quantity(line)
        return result

    def _try_western(self, line: str) -> ExtractedIngredient | None:
        """Try Western-style ingredient pattern: '200g chicken breast, diced'."""
        match = _RE_INGREDIENT_WESTERN.match(line.strip())
        if not match:
            return None

        quantity_str = match.group(1)
        unit = (match.group(2) or "").lower()
        rest = match.group(3).strip()

        # Split name and preparation
        name, prep = self._split_name_prep(rest)

        # Validate: name should be non-empty and look like a food item
        if not name or len(name) < 2:
            return None

        # Map unit to canonical form
        unit = _normalise_unit(unit) if unit else "piece"

        return ExtractedIngredient(
            raw_text=line.strip(),
            name=name.strip(),
            quantity=Decimal(quantity_str),
            unit=unit,
            preparation=prep.strip() if prep else None,
            extraction_source="EXPLICIT",
            confidence=Decimal("0.9"),
        )

    def _try_chinese(self, line: str) -> ExtractedIngredient | None:
        """Try Chinese-style ingredient pattern: '鸡胸肉 200g，切丁' or '大蒜3-4瓣'."""
        match = _RE_INGREDIENT_CHINESE.match(line.strip())
        if not match:
            return None

        name = match.group(1).strip()
        quantity_lo = match.group(2)
        quantity_hi = match.group(3)
        unit_raw = match.group(4)
        prep_raw = match.group(5)

        # Validate name
        if not name or len(name) < 1:
            return None

        # Map Chinese units
        unit = _normalise_unit(unit_raw.strip()) if unit_raw else "piece"

        # Quantity ranges ("3-4瓣") take the upper bound — conservative so the
        # plan never under-supplies (mirrors serving-range handling).
        quantity = Decimal(quantity_hi) if quantity_hi else Decimal(quantity_lo)

        # Clean preparation
        prep = prep_raw.strip() if prep_raw else None
        if prep:
            prep = _RE_STEP_NUMBER.sub("", prep).strip()  # Remove stray step numbers

        return ExtractedIngredient(
            raw_text=line.strip(),
            name=name,
            quantity=quantity,
            unit=unit,
            preparation=prep or None,
            extraction_source="EXPLICIT",
            confidence=Decimal("0.9"),
        )

    def _try_no_quantity(self, line: str) -> ExtractedIngredient | None:
        """Handle ingredients with no quantity: 'salt to taste', '适量盐', '老抽少许'."""
        stripped = line.strip()
        if not stripped:
            return None

        # Clean name noise BEFORE classification: parenthetical notes
        # ("味精/鸡精（可选）", "小米辣（依吃辣程度放）"), trailing qualifiers
        # ("老抽少许"), and trailing punctuation ("白胡椒粉、").
        cleaned = self._clean_ingredient_name(stripped)
        if not cleaned:
            return None

        # Any quantity qualifier anywhere in the line ("适量", "少许", ...)
        # marks this as a no-quantity ingredient line.
        if _RE_NO_QUANTITY.search(stripped) or _RE_QUANTITY_QUALIFIER.search(stripped):
            return ExtractedIngredient(
                raw_text=stripped,
                name=cleaned,
                extraction_source="EXPLICIT",
                confidence=Decimal("0.7"),
            )

        # Short text (likely a bare ingredient name) that doesn't read like a
        # step instruction → free-text ingredient. Classification runs on the
        # cleaned name so parenthetical notes never trigger step indicators.
        if len(cleaned) < 60 and not _RE_STEP_NUMBER.match(stripped):
            step_indicators = (
                "heat",
                "add",
                "mix",
                "stir",
                "cook",
                "bake",
                "boil",
                "fry",
                "热",
                "加",
                "放",
                "炒",
                "煮",
                "烤",
                "蒸",
                "拌",
            )
            lower = cleaned.lower()
            if any(ind in lower for ind in step_indicators):
                return None

            return ExtractedIngredient(
                raw_text=stripped,
                name=cleaned,
                extraction_source="EXPLICIT",
                confidence=Decimal("0.6"),
            )

        return None

    @staticmethod
    def _expand_slash_alternatives(ingredient: ExtractedIngredient) -> tuple[ExtractedIngredient, ...]:
        """Expand "味精/鸡精" style alternatives into separate ingredients.

        A slash in a no-quantity Chinese ingredient line usually means "or"
        ("味精/鸡精" = MSG or chicken powder). Splitting into one demand per
        alternative lets each match inventory independently; a lone slash with
        no CJK either side (e.g. "A/B sauce") is left untouched.
        """
        name = ingredient.name
        if "/" not in name:
            return (ingredient,)

        parts = [part.strip() for part in name.split("/") if part.strip()]
        if len(parts) < 2 or not all(any("\u4e00" <= ch <= "\u9fff" for ch in part) for part in parts):
            return (ingredient,)

        return tuple(
            ExtractedIngredient(
                raw_text=ingredient.raw_text,
                name=part,
                quantity=ingredient.quantity,
                unit=ingredient.unit,
                preparation=ingredient.preparation,
                extraction_source=ingredient.extraction_source,
                confidence=ingredient.confidence,
            )
            for part in parts
        )

    @staticmethod
    def _clean_ingredient_name(text: str) -> str:
        """Strip parenthetical notes, trailing qualifiers, and trailing punctuation.

        Applied to no-quantity ingredient lines so downstream inventory
        matching sees clean canonical names:
          "味精/鸡精（可选）"  → "味精/鸡精"
          "小米辣（依吃辣程度放）" → "小米辣"
          "老抽少许"           → "老抽"
          "白胡椒粉、"          → "白胡椒粉"
        """
        name = _RE_PAREN_NOTE.sub("", text).strip()
        name = _RE_TRAILING_QUANTITY.sub("", name).strip()
        name = _RE_TRAILING_PUNCT.sub("", name).strip()
        return name

    @staticmethod
    def _split_name_prep(text: str) -> tuple[str, str | None]:
        """Split 'chicken breast, diced' into (name, prep)."""
        match = _RE_NAME_PREP_SPLIT.search(text)
        if not match:
            return text.strip(), None

        name_part = text[: match.start()].strip()
        prep_part = match.group(1).strip()

        # Only treat as preparation if it contains a known prep keyword
        prep_lower = prep_part.lower()
        if any(kw in prep_lower for kw in _PREP_KEYWORDS):
            return name_part, prep_part

        # If the "prep" part is very short and doesn't look like prep,
        # it might be part of the name
        return text.strip(), None

    # ------------------------------------------------------------------
    # Step parsing
    # ------------------------------------------------------------------

    def _parse_step(self, index: int, line: str) -> ExtractedStep:
        """Parse a step line into ExtractedStep."""
        # Remove step number prefix for cleaner instruction text
        instruction = _RE_STEP_NUMBER.sub("", line).strip()

        # Detect cooking technique
        technique = self._detect_technique(line)

        # Map technique to category
        category = self._infer_category(technique)

        # Detect heat level
        heat = self._detect_heat(line)

        # Detect durations (active and passive)
        active_dur, passive_dur = self._detect_durations(line, technique)

        # Detect temperature
        temp = self._detect_temperature(line)

        # Detect resource hints
        resources = self._detect_resources(line)

        return ExtractedStep(
            step_number=index,
            instruction=instruction,
            category=category,
            active_duration_minutes=active_dur,
            passive_duration_minutes=passive_dur,
            heat_level=heat,
            target_temperature_c=temp,
            resources_hint=tuple(resources),
            extraction_source="EXPLICIT",
            confidence=Decimal("0.85"),
        )

    @staticmethod
    def _detect_technique(text: str) -> str:
        """Detect primary cooking technique from step text."""
        text_lower = text.lower()
        for technique, en_pat, zh_pat in _TECHNIQUE_PATTERNS:
            if re.search(en_pat, text_lower) or re.search(zh_pat, text):
                return technique
        return "general"

    @staticmethod
    def _infer_category(technique: str) -> str:
        """Map technique to step category."""
        heating_techniques = {
            "stir_fry",
            "deep_fry",
            "boil",
            "simmer",
            "steam",
            "bake",
            "roast",
            "grill",
            "sauté",
            "sear",
            "braise",
            "poach",
            "heat",
        }
        prep_techniques = {"marinate"}

        if technique in heating_techniques:
            return "heating"
        if technique in prep_techniques:
            return "preparation"
        return "general"

    @staticmethod
    def _detect_heat(text: str) -> HeatLevel:
        """Detect heat level from step text."""
        text_lower = text.lower()
        for pat, level in _HEAT_LEVELS:
            if re.search(pat, text_lower):
                return level
        return HeatLevel.NONE

    @staticmethod
    def _detect_durations(text: str, technique: str) -> tuple[int | None, int | None]:
        """Extract active and passive durations from step text."""
        # Try range: "10-15 minutes", "3~5分钟"
        match = _RE_DURATION_RANGE.search(text)
        if match:
            _lo, hi = int(match.group(1)), int(match.group(2))
            # For passive techniques (boil, simmer, bake, roast), duration is passive
            passive_techniques = {"boil", "simmer", "bake", "roast", "marinate", "steam", "braise"}
            if technique in passive_techniques:
                return None, hi  # Range max is passive, no explicit active
            return hi, None  # Active technique: treat as active duration

        # Try single: "10 minutes", "5分钟"
        match = _RE_DURATION_SINGLE.search(text)
        if match:
            minutes = int(match.group(1))
            passive_techniques = {"boil", "simmer", "bake", "roast", "marinate", "steam", "braise"}
            if technique in passive_techniques:
                return None, minutes
            return minutes, None

        return None, None

    @staticmethod
    def _detect_temperature(text: str) -> Decimal | None:
        """Detect target temperature from step text."""
        # Celsius: "180°C", "180 C", "200度"
        match = re.search(r"(\d{2,3})\s*(?:°\s*)?[cC](?:elsius)?\b", text)
        if match:
            return Decimal(match.group(1))

        match = re.search(r"(\d{2,3})\s*度\b", text)
        if match:
            return Decimal(match.group(1))

        # Fahrenheit: "350°F" → Celsius
        match = re.search(r"(\d{2,4})\s*(?:°\s*)?[fF](?:ahrenheit)?\b", text)
        if match:
            f = Decimal(match.group(1))
            return ((f - 32) * 5 / 9).quantize(Decimal("0.1"))

        return None

    @staticmethod
    def _detect_resources(text: str) -> list[str]:
        """Detect required kitchen resources from step text."""
        text_lower = text.lower()
        found: list[str] = []
        for resource, pattern in _RESOURCE_KEYWORDS.items():
            if re.search(pattern, text_lower):
                found.append(resource)
        return found

    # ------------------------------------------------------------------
    # Dish name extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dish_name(lines: list[str]) -> str:
        """Extract dish name from the first non-empty, non-section-header line."""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip section headers
            lower = stripped.lower()
            is_header = False
            for pat in _INGREDIENT_HEADER_PATTERNS:
                if pat.match(lower):
                    is_header = True
                    break
            for pat in _STEP_HEADER_PATTERNS:
                if pat.match(lower):
                    is_header = True
                    break
            if not is_header and not _RE_STEP_NUMBER.match(stripped):
                return stripped[:80]  # Truncate long names
        return "Untitled Recipe"

    @staticmethod
    def _make_recipe_id(dish_name: str) -> str:
        """Generate a stable recipe_id from dish name."""
        # Lowercase, replace spaces/special chars with underscores
        slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", dish_name.lower())
        return f"recipe_{slug[:40]}"

    @staticmethod
    def _detect_language(text: str) -> str:
        """Quick language detection for the candidate."""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        if cjk == 0 and latin == 0:
            return "und"
        if cjk > latin:
            return "zho"
        return "eng"

    @staticmethod
    def _extract_servings(text: str) -> Decimal:
        """Extract serving count from recipe text. Defaults to 2."""
        # "Serves 4", "4 servings", "2人份", "4人"
        match = re.search(r"(?:serves?|servings?|yields?|makes?)\s+(\d+)", text, re.IGNORECASE)
        if match:
            return Decimal(match.group(1))
        match = re.search(r"(\d+)\s*(?:人份|人份量| servings?)", text)
        if match:
            return Decimal(match.group(1))
        # "2-4 servings" — take the middle
        match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*(?:servings?|人份?)", text, re.IGNORECASE)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            return Decimal((lo + hi) // 2)
        return Decimal(2)


# =============================================================================
# Helper: unit normalisation
# =============================================================================


def _normalise_unit(unit: str) -> str:
    """Normalise a unit string to its canonical form."""
    unit_lower = unit.lower().strip()
    if unit_lower in _CHINESE_UNIT_MAP:
        return _CHINESE_UNIT_MAP[unit_lower]
    # Common English abbreviations
    unit_map = {
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "cup": "cup",
        "cups": "cup",
        "ounce": "oz",
        "ounces": "oz",
        "pound": "lb",
        "pounds": "lb",
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "milliliter": "ml",
        "milliliters": "ml",
        "litre": "l",
        "liter": "l",
        "litres": "l",
        "liters": "l",
    }
    return unit_map.get(unit_lower, unit_lower)
