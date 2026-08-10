"""Unit tests for ingredient name normalisation and catalogue matching."""

from __future__ import annotations

from cooking_plan_agent.normalisation.names import (
    clean_dish_name,
    match_catalogue_item,
    normalise_essential_resource,
    normalise_ingredient_name,
    normalise_resource_type,
)


class TestCleanDishName:
    """Dish display titles — quantities/notes/prep stripped, aliases NOT applied."""

    def test_plain_title_preserved(self):
        assert clean_dish_name("Crab legs") == "Crab legs"
        assert clean_dish_name("Salt and Pepper Chicken") == "Salt and Pepper Chicken"

    def test_parenthetical_notes_stripped(self):
        assert clean_dish_name("Fresh shrimp (， remove head， tail， and thread， and cut in half)") == "Fresh shrimp"
        assert clean_dish_name("Large-sized prawns 500 grams (select larger ones)") == "Large-sized prawns"

    def test_trailing_and_leading_quantities_stripped(self):
        assert clean_dish_name("Large-sized prawns 500 grams") == "Large-sized prawns"
        assert clean_dish_name("15 chicken wings") == "chicken wings"

    def test_prep_suffix_stripped(self):
        assert clean_dish_name("Chicken Wings, Fried") == "Chicken Wings"

    def test_untitled_placeholder_kept(self):
        # "Untitled Recipe" is handled by callers; the cleaner must not drop it.
        assert clean_dish_name("Untitled Recipe") == "Untitled Recipe"

    def test_empty_input_returns_cleaned_form(self):
        assert clean_dish_name("   ") == ""


class TestNormaliseIngredientName:
    """Ingredient name → canonical form."""

    def test_simple_name_preserved(self):
        """Plain ingredient names pass through unchanged (lowercased)."""
        assert normalise_ingredient_name("Salt") == "salt"
        assert normalise_ingredient_name("Rice") == "rice"
        assert normalise_ingredient_name("Garlic") == "garlic"

    def test_alias_resolved(self):
        """Known aliases map to canonical names."""
        assert normalise_ingredient_name("Chicken Breast Fillet") == "chicken breast"
        assert normalise_ingredient_name("green onion") == "spring onion"
        assert normalise_ingredient_name("scallion") == "spring onion"
        assert normalise_ingredient_name("capsicum") == "bell pepper"

    def test_chinese_aliases_resolved(self):
        """Chinese ingredient names share the English inventory key."""
        assert normalise_ingredient_name("鸡胸肉") == "chicken breast"
        assert normalise_ingredient_name("西红柿") == "tomato"
        assert normalise_ingredient_name("生抽") == "soy sauce"

    def test_quantity_prefix_stripped(self):
        """Quantity+unit prefixes are removed."""
        assert normalise_ingredient_name("200g chicken breast") == "chicken breast"
        assert normalise_ingredient_name("  500g Beef Sirloin  ") == "beef sirloin"

    def test_parenthetical_notes_stripped(self):
        """Content in parentheses is removed."""
        assert normalise_ingredient_name("chicken breast (fresh)") == "chicken breast"
        assert normalise_ingredient_name("tomato（medium）") == "tomato"

    def test_preparation_suffix_stripped(self):
        """Preparation notes after commas are removed."""
        assert normalise_ingredient_name("onion, diced") == "onion"
        assert normalise_ingredient_name("garlic，minced") == "garlic"

    def test_partial_alias_substring_match(self):
        """Alias appearing as substring triggers match."""
        assert normalise_ingredient_name("boneless skinless chicken breast") == "chicken breast"
        assert normalise_ingredient_name("unsalted butter") == "butter"

    def test_unknown_name_preserved(self):
        """Unknown names return cleaned lowercase form."""
        assert normalise_ingredient_name("saffron") == "saffron"
        assert normalise_ingredient_name("star anise") == "star anise"

    def test_whitespace_normalised(self):
        """Multiple spaces collapsed."""
        assert normalise_ingredient_name("chicken    breast") == "chicken breast"

    def test_longest_alias_wins(self):
        """When multiple partial aliases match, the one with most coverage wins."""
        # "chicken breast fillet" contains both "chicken" and "chicken breast"
        # "chicken breast" should win because it covers more of the name
        assert normalise_ingredient_name("Chicken Breast Fillet") == "chicken breast"


class TestMatchCatalogueItem:
    """Catalogue matching against an ingredient catalogue."""

    _CATALOGUE: dict[str, str] = {
        "chicken breast": "poultry_001",
        "beef sirloin": "beef_002",
        "salt": "seasoning_001",
        "rice": "grain_001",
        "garlic": "aromatic_001",
        "onion": "veg_001",
        "tomato": "veg_002",
    }

    def test_exact_match(self):
        """Normalised name exactly matches a catalogue key."""
        result = match_catalogue_item("salt", self._CATALOGUE)
        assert result.canonical_name == "salt"
        assert result.catalogue_id == "seasoning_001"
        assert result.confidence == 1.0
        assert result.matched_by == "exact"

    def test_alias_match(self):
        """Normalised name resolves via alias then matches catalogue exactly."""
        # "Chicken Breast Fillet" → normalise → "chicken breast" → exact match
        result = match_catalogue_item("Chicken Breast Fillet", self._CATALOGUE)
        assert result.canonical_name == "chicken breast"
        assert result.catalogue_id == "poultry_001"
        # Normalisation resolved the alias → exact catalogue match
        assert result.matched_by == "exact"
        assert result.confidence == 1.0

    def test_partial_token_match(self):
        """Normalised name matches catalogue key exactly."""
        cat_with_tomato = {"tomato": "veg_002", "cherry tomato": "veg_003"}
        # "cherry tomato fresh" → normalise → "cherry tomato" → exact match
        result = match_catalogue_item("cherry tomato fresh", cat_with_tomato)
        assert result.canonical_name == "cherry tomato"
        assert result.matched_by == "exact"

    def test_no_match(self):
        """Unknown ingredient returns empty match."""
        result = match_catalogue_item("saffron", self._CATALOGUE)
        assert result.canonical_name == "saffron"  # from normalisation
        assert result.catalogue_id == ""
        assert result.confidence == 0.0
        assert result.matched_by == "none"

    def test_multiple_matches_picks_best_confidence(self):
        """When multiple catalogue keys partially match, the highest confidence wins."""
        cat = {
            "chicken": "poultry_generic",
            "chicken breast": "poultry_specific",
            "chicken thigh": "poultry_alt",
        }
        result = match_catalogue_item("chicken breast fillet", cat)
        assert result.canonical_name == "chicken breast"
        assert result.catalogue_id == "poultry_specific"


class TestNormaliseResourceType:
    def test_bilingual_resource_aliases_resolved(self):
        assert normalise_resource_type("燃气灶") == "stove"
        assert normalise_resource_type("炒锅") == "wok"
        assert normalise_resource_type("cutting-board") == "cutting_board"

    def test_llm_noisy_labels_resolved_to_canonical(self):
        """LLM 自由文本的器材标签也需归一化为稳定类型。"""
        assert normalise_resource_type("锅") == "pot"
        assert normalise_resource_type("电饭煲") == "rice_cooker"
        assert normalise_resource_type("rice cooker") == "rice_cooker"
        assert normalise_resource_type("碗") == "mixing_bowl"

    def test_unknown_label_preserved(self):
        """未知标签保留归一化形式，供自定义类型精确匹配。"""
        assert normalise_resource_type("剪刀") == "剪刀"
        assert normalise_resource_type("厨房纸") == "厨房纸"
        assert normalise_resource_type("锅盖") == "锅盖"


class TestNormaliseEssentialResource:
    def test_essential_equipment_resolved(self):
        """关键器材（能力定义型）返回规范类型。"""
        assert normalise_essential_resource("stove") == "stove"
        assert normalise_essential_resource("燃气灶") == "stove"
        assert normalise_essential_resource("锅") == "pot"
        assert normalise_essential_resource("菜刀") == "knife"
        assert normalise_essential_resource("oven") == "oven"

    def test_capability_suffix_stripped(self):
        """'stove:induction' → 'stove'（与 repair 目录查询保持一致）。"""
        assert normalise_essential_resource("stove:induction") == "stove"

    def test_soft_or_unknown_hints_return_none(self):
        """耗材/容器/配件/可替代家电/未知标签不参与可行性 gating。"""
        assert normalise_essential_resource("剪刀") is None
        assert normalise_essential_resource("厨房纸") is None
        assert normalise_essential_resource("碗") is None
        assert normalise_essential_resource("锅盖") is None
        assert normalise_essential_resource("电饭煲") is None  # 可替代家电，不做 gating
        assert normalise_essential_resource("laser_cutter") is None
