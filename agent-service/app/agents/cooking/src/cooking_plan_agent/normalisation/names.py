"""Ingredient name normalisation and catalogue matching.

Handbook 5.1–5.2: normalise raw ingredient names to canonical forms and
match them against a curated ingredient catalogue.  This is the first
step of the canonicalisation pipeline — unit normalisation follows in
``units.py``.

Design: all functions are pure (no I/O).  The catalogue is injected as a
parameter so the module remains testable without a database.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

# =============================================================================
# 5.1  Ingredient name normalisation
# =============================================================================

# Common ingredient aliases — maps raw names to canonical names.
# Order: longer patterns are checked first to avoid partial matches.
_INGREDIENT_ALIASES: list[tuple[str, str]] = [
    # --- Poultry ---
    ("chicken breast fillet", "chicken breast"),
    ("chicken thigh fillet", "chicken thigh"),
    ("boneless skinless chicken breast", "chicken breast"),
    ("boneless chicken thigh", "chicken thigh"),
    ("chicken drumstick", "chicken drumstick"),
    ("chicken wing", "chicken wing"),
    # --- Beef ---
    ("beef sirloin steak", "beef sirloin"),
    ("beef ribeye steak", "beef ribeye"),
    ("beef tenderloin", "beef fillet"),
    ("ground beef", "minced beef"),
    ("beef mince", "minced beef"),
    # --- Pork ---
    ("pork belly slice", "pork belly"),
    ("pork shoulder", "pork shoulder"),
    ("ground pork", "minced pork"),
    ("pork mince", "minced pork"),
    # --- Seafood ---
    ("prawn", "shrimp"),
    ("large shrimp", "shrimp"),
    ("jumbo shrimp", "shrimp"),
    ("salmon fillet", "salmon"),
    # --- Vegetables ---
    ("white onion", "onion"),
    ("brown onion", "onion"),
    ("yellow onion", "onion"),
    ("red onion", "red onion"),
    ("spring onion", "spring onion"),
    ("green onion", "spring onion"),
    ("scallion", "spring onion"),
    ("bell pepper", "bell pepper"),
    ("capsicum", "bell pepper"),
    ("chilli pepper", "chilli"),
    ("chili pepper", "chilli"),
    ("red chilli", "chilli"),
    ("green chilli", "chilli"),
    ("tomato vine", "tomato"),
    ("cherry tomato", "cherry tomato"),
    ("romaine lettuce", "lettuce"),
    ("iceberg lettuce", "lettuce"),
    ("cabbage", "cabbage"),
    ("chinese cabbage", "napa cabbage"),
    ("napa cabbage", "napa cabbage"),
    ("broccoli floret", "broccoli"),
    ("cauliflower floret", "cauliflower"),
    # --- Grains & staples ---
    ("jasmine rice", "rice"),
    ("basmati rice", "rice"),
    ("white rice", "rice"),
    ("brown rice", "brown rice"),
    ("wheat flour", "flour"),
    ("all-purpose flour", "flour"),
    ("plain flour", "flour"),
    ("corn starch", "cornstarch"),
    ("cornflour", "cornstarch"),
    # --- Dairy ---
    ("whole milk", "milk"),
    ("skim milk", "milk"),
    ("unsalted butter", "butter"),
    ("salted butter", "butter"),
    ("heavy cream", "cream"),
    ("double cream", "cream"),
    ("whipping cream", "cream"),
    # --- Condiments & sauces ---
    ("light soy sauce", "soy sauce"),
    ("dark soy sauce", "soy sauce"),
    ("fish sauce", "fish sauce"),
    ("oyster sauce", "oyster sauce"),
    ("sesame oil", "sesame oil"),
    ("olive oil", "olive oil"),
    ("vegetable oil", "vegetable oil"),
    ("cooking oil", "vegetable oil"),
    ("rice vinegar", "rice vinegar"),
    ("white vinegar", "vinegar"),
    # --- Seasonings ---
    ("white pepper", "white pepper"),
    ("black pepper", "black pepper"),
    ("ground pepper", "black pepper"),
    ("sea salt", "salt"),
    ("table salt", "salt"),
    ("kosher salt", "salt"),
    ("rock salt", "salt"),
    ("white sugar", "sugar"),
    ("brown sugar", "brown sugar"),
    ("caster sugar", "sugar"),
    ("granulated sugar", "sugar"),
    # --- Aromatics ---
    ("garlic clove", "garlic"),
    ("garlic bulb", "garlic"),
    ("ginger root", "ginger"),
    ("fresh ginger", "ginger"),
    ("coriander", "cilantro"),
    # --- Chinese LLM / user input aliases ---
    ("鸡胸肉", "chicken breast"),
    ("鸡胸", "chicken breast"),
    ("鸡腿肉", "chicken thigh"),
    ("鸡腿", "chicken thigh"),
    ("牛里脊", "beef fillet"),
    ("牛肉末", "minced beef"),
    ("猪肉末", "minced pork"),
    ("五花肉", "pork belly"),
    ("虾", "shrimp"),
    ("大虾", "shrimp"),
    ("三文鱼", "salmon"),
    ("洋葱", "onion"),
    ("红洋葱", "red onion"),
    ("葱", "spring onion"),
    ("青葱", "spring onion"),
    ("小葱", "spring onion"),
    ("彩椒", "bell pepper"),
    ("甜椒", "bell pepper"),
    ("辣椒", "chilli"),
    ("番茄", "tomato"),
    ("西红柿", "tomato"),
    ("小番茄", "cherry tomato"),
    ("生菜", "lettuce"),
    ("卷心菜", "cabbage"),
    ("包菜", "cabbage"),
    ("大白菜", "napa cabbage"),
    ("西兰花", "broccoli"),
    ("花椰菜", "cauliflower"),
    ("米", "rice"),
    ("大米", "rice"),
    ("面粉", "flour"),
    ("玉米淀粉", "cornstarch"),
    ("牛奶", "milk"),
    ("黄油", "butter"),
    ("淡奶油", "cream"),
    ("酱油", "soy sauce"),
    ("生抽", "soy sauce"),
    ("老抽", "soy sauce"),
    ("蚝油", "oyster sauce"),
    ("鱼露", "fish sauce"),
    ("香油", "sesame oil"),
    ("食用油", "vegetable oil"),
    ("盐", "salt"),
    ("白糖", "sugar"),
    ("蒜", "garlic"),
    ("大蒜", "garlic"),
    ("姜", "ginger"),
    ("香菜", "cilantro"),
]

# Kitchen resources use the same internal vocabulary as ResourceNeed and
# KitchenResourceSnapshot.  LLM output is allowed to be bilingual, but
# comparisons must always use these stable keys.
_RESOURCE_ALIASES: dict[str, str] = {
    "stove": "stove",
    "gas stove": "stove",
    "cooktop": "stove",
    "hob": "stove",
    "burner": "stove",
    "炉灶": "stove",
    "燃气灶": "stove",
    "电磁炉": "stove",
    "oven": "oven",
    "烤箱": "oven",
    "wok": "wok",
    "炒锅": "wok",
    "frying pan": "frying_pan",
    "skillet": "frying_pan",
    "平底锅": "frying_pan",
    "煎锅": "frying_pan",
    "pot": "pot",
    "saucepan": "pot",
    "汤锅": "pot",
    "煮锅": "pot",
    "knife": "knife",
    "chef knife": "knife",
    "菜刀": "knife",
    "厨刀": "knife",
    "cutting board": "cutting_board",
    "chopping board": "cutting_board",
    "砧板": "cutting_board",
    "mixing bowl": "mixing_bowl",
    "搅拌碗": "mixing_bowl",
    "调料碗": "mixing_bowl",
    "sink": "sink",
    "水槽": "sink",
    "洗菜池": "sink",
    "spatula": "spatula",
    "turner": "spatula",
    "锅铲": "spatula",
    "铲子": "spatula",
    "steamer": "steamer",
    "蒸锅": "steamer",
    "microwave": "microwave",
    "microwave oven": "microwave",
    "微波炉": "microwave",
    "blender": "blender",
    "food processor": "blender",
    "料理机": "blender",
    "搅拌机": "blender",
    "air fryer": "air_fryer",
    "空气炸锅": "air_fryer",
    "rice cooker": "rice_cooker",
    "电饭煲": "rice_cooker",
    "电饭锅": "rice_cooker",
    "锅": "pot",
    "碗": "mixing_bowl",
    "菜板": "cutting_board",
    "案板": "cutting_board",
}

# Resource types that the pipeline itself plans around — they appear in the
# technique→resource inference table, the decomposition ResourceNeed set, or
# the rule-based keyword extractor. The feasibility gate uses ONLY these: a
# resources_hint that does not resolve to one of them — consumables (厨房纸),
# containers (碗), accessories (锅盖), substitutable appliances (电饭煲), or
# unknown LLM labels (剪刀) — is a soft hint and can never make a plan
# infeasible on its own.  The per-task canonical ResourceNeed produced by
# decomposition remains the authoritative scheduling contract.
ESSENTIAL_RESOURCE_TYPES: frozenset[str] = frozenset(
    {
        "stove",
        "oven",
        "wok",
        "frying_pan",
        "pan",
        "pot",
        "knife",
        "cutting_board",
        "sink",
        "spatula",
        "steamer",
    }
)

# Patterns to strip from ingredient names during normalisation.
# Applied BEFORE alias mapping — removes quantity, unit, and prep notes
# that clung to the ingredient name during extraction.
_RE_STRIP_QUANTITY = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:g|kg|ml|l|tbsp|tsp|cup|cups|oz|lb|piece|pc|pcs)\s+",
    re.IGNORECASE,
)
_RE_STRIP_PREP = re.compile(
    r"\s*[，,]\s*(?:diced|minced|chopped|sliced|julienned|crushed|peeled|grated|cut|"
    r"fried|battered|breaded|marinated|seared|grilled|roasted|steamed|stir[\s-]?fried|pan[\s-]?fried).*$",
    re.IGNORECASE,
)
_RE_STRIP_PAREN = re.compile(r"\s*[（(][^)）]*[)）]\s*")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")

# Dish-name cleaning (display titles, not ingredients): strip quantities,
# parenthetical notes, and preparation suffixes that cling to the name.
_RE_DISH_TRAILING_QUANTITY = re.compile(
    r"\s+\d+(?:\.\d+)?\s*(?:g|kg|ml|l|tbsp|tsp|cup|cups|oz|lb|grams?|kilograms?|"
    r"milliliters?|piece|pieces?|pc|pcs|cloves?|slices?|bunch|bunches)\s*$",
    re.IGNORECASE,
)
_RE_DISH_LEADING_QUANTITY = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:x\s*)?(?=[A-Za-z])")


def clean_dish_name(raw_name: str) -> str:
    """Normalise a raw dish name to a short display title.

    Pipeline:
      1. Unicode normalise and strip.
      2. Remove parenthetical notes (e.g. "(remove head and tail)").
      3. Remove preparation suffixes (e.g. ", deep-fried").
      4. Remove trailing/leading quantities and units (e.g. "500 grams").
      5. Collapse whitespace.

    Ingredient catalogue aliases are NOT applied here — a dish name is
    user-facing ("Large-Sized Prawns") and must stay close to the source.

    Examples:
        >>> clean_dish_name('Fresh shrimp (， remove head， tail， and thread， and cut in half)')
        'Fresh shrimp'
        >>> clean_dish_name('Large-sized prawns 500 grams (select larger ones)')
        'Large-sized prawns'
        >>> clean_dish_name('15 chicken wings')
        'chicken wings'
        >>> clean_dish_name('Salt and Pepper Chicken')
        'Salt and Pepper Chicken'
    """
    name = unicodedata.normalize("NFKC", raw_name).strip()
    name = _RE_STRIP_PAREN.sub(" ", name)
    name = _RE_STRIP_PREP.sub("", name)
    name = _RE_DISH_TRAILING_QUANTITY.sub("", name)
    name = _RE_DISH_LEADING_QUANTITY.sub("", name)
    name = _RE_MULTI_SPACE.sub(" ", name).strip()
    return name or raw_name.strip()


def normalise_ingredient_name(raw_name: str) -> str:
    """Normalise a raw ingredient name to its canonical form.

    Pipeline:
      1. Strip whitespace and lowercase.
      2. Remove quantity prefixes and parenthetical notes.
      3. Remove preparation suffixes.
      4. Apply alias mapping (longest-pattern-first).
      5. Collapse multiple spaces.

    If no alias matches, the cleaned name is returned as-is — the caller
    can then attempt catalogue matching.

    Args:
        raw_name: Raw ingredient name from recipe text or user input.

    Returns:
        Canonical ingredient name (lowercase, trimmed, de-aliased).

    Examples:
        >>> normalise_ingredient_name('Chicken Breast Fillet')
        'chicken breast'
        >>> normalise_ingredient_name('  200g Boneless Skinless Chicken Breast  ')
        'chicken breast'
        >>> normalise_ingredient_name('Salt')
        'salt'
    """
    name = unicodedata.normalize("NFKC", raw_name).strip().lower()

    # Stage 1: strip quantity + unit prefix (e.g. "200g chicken breast" → "chicken breast")
    name = _RE_STRIP_QUANTITY.sub("", name).strip()

    # Stage 2: strip parenthetical notes (e.g. "chicken (fresh)" → "chicken")
    name = _RE_STRIP_PAREN.sub(" ", name)

    # Stage 3: strip preparation suffixes (e.g. "onion, diced" → "onion")
    name = _RE_STRIP_PREP.sub("", name)

    # Stage 4: collapse whitespace
    name = _RE_MULTI_SPACE.sub(" ", name).strip()

    if not name:
        return raw_name.strip().lower()

    # Stage 5: alias mapping — check longest patterns first
    for alias, canonical in _INGREDIENT_ALIASES:
        if name == alias:
            return canonical

    # Partial alias matching: if the cleaned name starts with or contains a
    # known alias (e.g. "chicken breast" appears in "chicken breast fillet")
    # Only apply if the match covers > 60% of the cleaned name.
    for alias, canonical in _INGREDIENT_ALIASES:
        # CJK aliases are commonly two to four characters, whereas the
        # equivalent English aliases need a longer threshold to avoid noise.
        if (len(alias) > 4 or any("\u4e00" <= char <= "\u9fff" for char in alias)) and alias in name:
            return canonical

    return name


def normalise_resource_type(raw_type: str) -> str:
    """Normalise bilingual equipment labels to a stable resource type.

    Unknown labels are returned in a whitespace- and separator-normalised
    form so existing custom resource types continue to match exactly.
    """
    resource_type = unicodedata.normalize("NFKC", raw_type).strip().lower()
    resource_type = _RE_MULTI_SPACE.sub(" ", resource_type)
    resource_type = resource_type.replace("-", " ").replace("_", " ")
    resource_type = _RE_MULTI_SPACE.sub(" ", resource_type).strip()
    return _RESOURCE_ALIASES.get(resource_type, resource_type.replace(" ", "_"))


def normalise_essential_resource(raw_type: str) -> str | None:
    """Normalise a resource hint, returning its canonical type only when it
    maps to an essential equipment type.

    Hints that do not resolve to one of :data:`ESSENTIAL_RESOURCE_TYPES`
    (consumables, containers, accessories, or labels outside the canonical
    vocabulary) return ``None`` so the feasibility gate can ignore them —
    e.g. a cutting step whose text mentions 剪刀/厨房纸/碗 must not become
    infeasible when a knife is available.

    Args:
        raw_type: Raw resource label from a recipe hint or kitchen snapshot.

    Returns:
        Canonical essential type (capability suffix stripped), or None.
    """
    canonical = normalise_resource_type(raw_type)
    # Strip capability suffix (e.g. "stove:induction" → "stove").
    base = canonical.split(":", 1)[0].strip()
    if base in ESSENTIAL_RESOURCE_TYPES:
        return base
    return None


# =============================================================================
# 5.2  Catalogue item matching
# =============================================================================


class CanonicalIngredientMatch(NamedTuple):
    """Result of matching a raw ingredient name against a catalogue.

    Attributes:
        canonical_name: The matched catalogue entry's canonical name.
        catalogue_id: Identifier of the matched catalogue entry (empty if no match).
        confidence: Match confidence [0, 1]. 1.0 = exact match, < 0.5 = weak.
        matched_by: How the match was found: "exact" | "alias" | "partial" | "none".
    """

    canonical_name: str
    catalogue_id: str
    confidence: float
    matched_by: str


# Minimal catalogue entry for MVP — production systems replace this with
# a database-backed ingredient catalogue.
IngredientCatalogue = dict[str, str]
"""Catalogue mapping: canonical_name → catalogue_id.

Example:
    {"chicken breast": "cat_poultry_001", "salt": "cat_seasoning_042"}
"""


def match_catalogue_item(
    raw_name: str,
    catalogue: IngredientCatalogue,
) -> CanonicalIngredientMatch:
    """Match a (possibly raw) ingredient name against an ingredient catalogue.

    Strategy:
      1. Normalise the name via ``normalise_ingredient_name``.
      2. Exact match against catalogue keys (case-insensitive).
      3. Alias-based match: check if any catalogue key is a substring.
      4. Partial match: check token overlap (Jaccard-like).  At least 50 %
         token overlap required.

    Args:
        raw_name: Raw ingredient name from recipe text.
        catalogue: A dict mapping canonical_name → catalogue_id.

    Returns:
        CanonicalIngredientMatch with the best match.  catalogue_id is
        empty and confidence is 0.0 when no match is found.

    Examples:
        >>> cat = {"chicken breast": "p001", "salt": "s001"}
        >>> match_catalogue_item("Chicken Breast Fillet", cat)
        CanonicalIngredientMatch(canonical_name='chicken breast', catalogue_id='p001', confidence=0.7, matched_by='alias')
        >>> match_catalogue_item("saffron", cat)
        CanonicalIngredientMatch(canonical_name='saffron', catalogue_id='', confidence=0.0, matched_by='none')
    """
    normalised = normalise_ingredient_name(raw_name)

    # --- Step 1: exact match ---
    if normalised in catalogue:
        return CanonicalIngredientMatch(
            canonical_name=normalised,
            catalogue_id=catalogue[normalised],
            confidence=1.0,
            matched_by="exact",
        )

    # --- Step 2: normalised-name-as-alias lookup ---
    # Try matching the normalised name against catalogue keys via alias mapping.
    # This catches cases where normalise_ingredient_name didn't resolve to a
    # catalogue key but the catalogue key is itself a broader category.
    for cat_key, cat_id in catalogue.items():
        if normalised == cat_key:
            return CanonicalIngredientMatch(
                canonical_name=cat_key,
                catalogue_id=cat_id,
                confidence=1.0,
                matched_by="exact",
            )

    # --- Step 3: substring / alias match ---
    # Check if any catalogue key appears as a substring of the normalised name.
    best_sub: CanonicalIngredientMatch | None = None
    for cat_key, cat_id in catalogue.items():
        if cat_key in normalised and len(cat_key) > 3:
            confidence = len(cat_key) / max(len(normalised), 1)
            if best_sub is None or confidence > best_sub.confidence:
                best_sub = CanonicalIngredientMatch(
                    canonical_name=cat_key,
                    catalogue_id=cat_id,
                    confidence=round(confidence, 2),
                    matched_by="alias",
                )

    if best_sub is not None and best_sub.confidence >= 0.5:
        return best_sub

    # --- Step 4: token-overlap partial match ---
    # Split both sides into tokens.  If >= 50 % of the raw name's tokens
    # appear in a catalogue key (or vice versa), accept as partial match.
    raw_tokens = set(normalised.split())
    best_partial: CanonicalIngredientMatch | None = None

    for cat_key, cat_id in catalogue.items():
        cat_tokens = set(cat_key.split())
        if not raw_tokens or not cat_tokens:
            continue
        overlap = len(raw_tokens & cat_tokens)
        jaccard = overlap / len(raw_tokens | cat_tokens)
        if jaccard >= 0.5:
            if best_partial is None or jaccard > best_partial.confidence:
                best_partial = CanonicalIngredientMatch(
                    canonical_name=cat_key,
                    catalogue_id=cat_id,
                    confidence=round(jaccard, 2),
                    matched_by="partial",
                )

    if best_partial is not None:
        return best_partial

    # --- No match ---
    return CanonicalIngredientMatch(
        canonical_name=normalised,
        catalogue_id="",
        confidence=0.0,
        matched_by="none",
    )
