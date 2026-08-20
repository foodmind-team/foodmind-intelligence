# =============================================================================
# 食材名称规范化与目录匹配模块（normalisation/names）
# -----------------------------------------------------------------------------
# 实现手册 5.1–5.2：把原始食材名规范化到规范形式，并与精选食材目录做匹配。
# 这是“规范化流水线”的第一步 —— 单位规范化随后在 units.py 中完成。
# 设计：所有函数都是纯函数（无 I/O）。目录作为参数注入，使模块无需数据库即可测试。
# 核心函数：
#   - clean_dish_name          ：清洗菜名（展示标题，非食材）
#   - normalise_ingredient_name：规范化食材名（去数量/括号/预处理后缀 + 别名映射）
#   - normalise_resource_type  ：规范化双语设备标签为稳定资源类型
#   - normalise_essential_resource：规范化资源提示，仅当映射到“必需设备”时返回规范类型
#   - match_catalogue_item     ：把食材名与目录匹配（精确 → 别名 → 部分 token 重叠）
# =============================================================================

"""Ingredient name normalisation and catalogue matching.

食材名称规范化与目录匹配。

Handbook 5.1–5.2: normalise raw ingredient names to canonical forms and
match them against a curated ingredient catalogue.  This is the first
step of the canonicalisation pipeline — unit normalisation follows in
``units.py``.

手册 5.1–5.2：把原始食材名规范化到规范形式，并与精选食材目录匹配。
这是规范化流水线的第一步 —— 单位规范化随后在 units.py 中完成。

Design: all functions are pure (no I/O).  The catalogue is injected as a
parameter so the module remains testable without a database.

设计：所有函数都是纯函数（无 I/O）。目录作为参数注入，使模块无需数据库即可测试。
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

# =============================================================================
# 5.1  Ingredient name normalisation
# 5.1  食材名称规范化
# =============================================================================

# Common ingredient aliases — maps raw names to canonical names.
# Order: longer patterns are checked first to avoid partial matches.
# 常见食材别名 —— 把原始名称映射到规范名称。顺序：较长的模式先检查，避免部分匹配。
_INGREDIENT_ALIASES: list[tuple[str, str]] = [
    # --- Poultry 禽类 ---
    ("chicken breast fillet", "chicken breast"),
    ("chicken thigh fillet", "chicken thigh"),
    ("boneless skinless chicken breast", "chicken breast"),
    ("boneless chicken thigh", "chicken thigh"),
    ("chicken drumstick", "chicken drumstick"),
    ("chicken wing", "chicken wing"),
    # --- Beef 牛肉 ---
    ("beef sirloin steak", "beef sirloin"),
    ("beef ribeye steak", "beef ribeye"),
    ("beef tenderloin", "beef fillet"),
    ("ground beef", "minced beef"),
    ("beef mince", "minced beef"),
    # --- Pork 猪肉 ---
    ("pork belly slice", "pork belly"),
    ("pork shoulder", "pork shoulder"),
    ("ground pork", "minced pork"),
    ("pork mince", "minced pork"),
    # --- Seafood 海鲜 ---
    ("prawn", "shrimp"),
    ("large shrimp", "shrimp"),
    ("jumbo shrimp", "shrimp"),
    ("salmon fillet", "salmon"),
    # --- Vegetables 蔬菜 ---
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
    # --- Grains & staples 谷物与主食 ---
    ("jasmine rice", "rice"),
    ("basmati rice", "rice"),
    ("white rice", "rice"),
    ("brown rice", "brown rice"),
    ("wheat flour", "flour"),
    ("all-purpose flour", "flour"),
    ("plain flour", "flour"),
    ("corn starch", "cornstarch"),
    ("cornflour", "cornstarch"),
    # --- Dairy 乳制品 ---
    ("whole milk", "milk"),
    ("skim milk", "milk"),
    ("unsalted butter", "butter"),
    ("salted butter", "butter"),
    ("heavy cream", "cream"),
    ("double cream", "cream"),
    ("whipping cream", "cream"),
    # --- Condiments & sauces 调味品与酱料 ---
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
    # --- Seasonings 调味料 ---
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
    # --- Aromatics 辛香料 ---
    ("garlic clove", "garlic"),
    ("garlic bulb", "garlic"),
    ("ginger root", "ginger"),
    ("fresh ginger", "ginger"),
    ("coriander", "cilantro"),
    # --- Chinese LLM / user input aliases 中文 LLM / 用户输入别名 ---
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
# 厨房资源与 ResourceNeed / KitchenResourceSnapshot 使用同一内部词汇。
# LLM 输出允许双语，但比较时必须使用这些稳定键。
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
# 流水线自身规划的资源类型 —— 出现在“技法→资源”推断表、分解 ResourceNeed 集
# 或基于规则的关键词提取器中。可行性闸门只使用这些类型：一个未解析到这些类型的
# resources_hint —— 消耗品（厨房纸）、容器（碗）、配件（锅盖）、可替换电器（电饭煲）、
# 或未知 LLM 标签（剪刀）—— 是软提示，绝不能单独使计划不可行。
# 分解产生的“每任务规范 ResourceNeed”才是权威调度契约。
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
# 规范化时从食材名中剥离的模式。在别名映射之前应用 —— 移除提取时黏在
# 食材名上的数量、单位与预处理备注。
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
# 菜名清洗（展示标题，非食材）：剥离数量、括号备注与黏在名字上的预处理后缀。
_RE_DISH_TRAILING_QUANTITY = re.compile(
    r"\s+\d+(?:\.\d+)?\s*(?:g|kg|ml|l|tbsp|tsp|cup|cups|oz|lb|grams?|kilograms?|"
    r"milliliters?|piece|pieces?|pc|pcs|cloves?|slices?|bunch|bunches)\s*$",
    re.IGNORECASE,
)
_RE_DISH_LEADING_QUANTITY = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:x\s*)?(?=[A-Za-z])")


def clean_dish_name(raw_name: str) -> str:
    """把原始菜名规范化为简短展示标题。

    Normalise a raw dish name to a short display title.

    Pipeline:
      1. Unicode normalise and strip.
      2. Remove parenthetical notes (e.g. "(remove head and tail)").
      3. Remove preparation suffixes (e.g. ", deep-fried").
      4. Remove trailing/leading quantities and units (e.g. "500 grams").
      5. Collapse whitespace.

    流水线：
      1. Unicode 规范化并去空白。
      2. 移除括号备注（如 "(remove head and tail)"）。
      3. 移除预处理后缀（如 ", deep-fried"）。
      4. 移除尾随 / 前导数量与单位（如 "500 grams"）。
      5. 折叠空白。

    Ingredient catalogue aliases are NOT applied here — a dish name is
    user-facing ("Large-Sized Prawns") and must stay close to the source.

    此处不应用食材目录别名 —— 菜名是面向用户的（"Large-Sized Prawns"），必须贴近原文。

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
    """把原始食材名规范化为规范形式。

    Normalise a raw ingredient name to its canonical form.

    Pipeline:
      1. Strip whitespace and lowercase.
      2. Remove quantity prefixes and parenthetical notes.
      3. Remove preparation suffixes.
      4. Apply alias mapping (longest-pattern-first).
      5. Collapse multiple spaces.

    流水线：
      1. 去空白并转小写。
      2. 移除数量前缀与括号备注。
      3. 移除预处理后缀。
      4. 应用别名映射（最长模式优先）。
      5. 折叠多个空格。

    If no alias matches, the cleaned name is returned as-is — the caller
    can then attempt catalogue matching.

    若无别名匹配，则原样返回清洗后的名称 —— 调用方可随后尝试目录匹配。

    Args:
        raw_name: Raw ingredient name from recipe text or user input.
            raw_name：来自菜谱文本或用户输入的原始食材名。

    Returns:
        Canonical ingredient name (lowercase, trimmed, de-aliased).
        规范食材名（小写、去空白、去别名）。

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
    # 第 1 阶段：剥离数量 + 单位前缀（如 "200g chicken breast" → "chicken breast"）
    name = _RE_STRIP_QUANTITY.sub("", name).strip()

    # Stage 2: strip parenthetical notes (e.g. "chicken (fresh)" → "chicken")
    # 第 2 阶段：剥离括号备注（如 "chicken (fresh)" → "chicken"）
    name = _RE_STRIP_PAREN.sub(" ", name)

    # Stage 3: strip preparation suffixes (e.g. "onion, diced" → "onion")
    # 第 3 阶段：剥离预处理后缀（如 "onion, diced" → "onion"）
    name = _RE_STRIP_PREP.sub("", name)

    # Stage 4: collapse whitespace
    # 第 4 阶段：折叠空白
    name = _RE_MULTI_SPACE.sub(" ", name).strip()

    if not name:
        return raw_name.strip().lower()

    # Stage 5: alias mapping — check longest patterns first
    # 第 5 阶段：别名映射 —— 先检查最长模式
    for alias, canonical in _INGREDIENT_ALIASES:
        if name == alias:
            return canonical

    # Partial alias matching: if the cleaned name starts with or contains a
    # known alias (e.g. "chicken breast" appears in "chicken breast fillet")
    # Only apply if the match covers > 60% of the cleaned name.
    # 部分别名匹配：若清洗后的名称以已知别名开头或包含已知别名
    # （如 "chicken breast" 出现在 "chicken breast fillet" 中）。
    # 仅当匹配覆盖清洗后名称的 > 60% 时才应用。
    for alias, canonical in _INGREDIENT_ALIASES:
        # CJK aliases are commonly two to four characters, whereas the
        # equivalent English aliases need a longer threshold to avoid noise.
        # CJK 别名通常二到四个字符，而对应英文别名需要更长阈值以避免噪声。
        if (len(alias) > 4 or any("\u4e00" <= char <= "\u9fff" for char in alias)) and alias in name:
            return canonical

    return name


def normalise_resource_type(raw_type: str) -> str:
    """把双语设备标签规范化为稳定资源类型。

    Normalise bilingual equipment labels to a stable resource type.

    Unknown labels are returned in a whitespace- and separator-normalised
    form so existing custom resource types continue to match exactly.

    未知标签以“空白与分隔符规范化后”的形式返回，使既有自定义资源类型仍能精确匹配。
    """
    resource_type = unicodedata.normalize("NFKC", raw_type).strip().lower()
    resource_type = _RE_MULTI_SPACE.sub(" ", resource_type)
    resource_type = resource_type.replace("-", " ").replace("_", " ")
    resource_type = _RE_MULTI_SPACE.sub(" ", resource_type).strip()
    return _RESOURCE_ALIASES.get(resource_type, resource_type.replace(" ", "_"))


def normalise_essential_resource(raw_type: str) -> str | None:
    """规范化资源提示，仅当映射到必需设备类型时返回其规范类型。

    Normalise a resource hint, returning its canonical type only when it
    maps to an essential equipment type.

    Hints that do not resolve to one of :data:`ESSENTIAL_RESOURCE_TYPES`
    (consumables, containers, accessories, or labels outside the canonical
    vocabulary) return ``None`` so the feasibility gate can ignore them —
    e.g. a cutting step whose text mentions 剪刀/厨房纸/碗 must not become
    infeasible when a knife is available.

    未解析到 ESSENTIAL_RESOURCE_TYPES 之一的提示（消耗品、容器、配件，或
    规范词汇之外的标签）返回 None，使可行性闸门可忽略它们 ——
    例如，一个切菜步骤文本提到 剪刀 / 厨房纸 / 碗 时，若已有刀可用，绝不能变得不可行。

    Args:
        raw_type: Raw resource label from a recipe hint or kitchen snapshot.
            raw_type：来自菜谱提示或厨房快照的原始资源标签。

    Returns:
        Canonical essential type (capability suffix stripped), or None.
        规范必需类型（剥离能力后缀），或 None。
    """
    canonical = normalise_resource_type(raw_type)
    # Strip capability suffix (e.g. "stove:induction" → "stove").
    # 剥离能力后缀（如 "stove:induction" → "stove"）。
    base = canonical.split(":", 1)[0].strip()
    if base in ESSENTIAL_RESOURCE_TYPES:
        return base
    return None


# =============================================================================
# 5.2  Catalogue item matching
# 5.2  目录条目匹配
# =============================================================================


class CanonicalIngredientMatch(NamedTuple):
    """把原始食材名与目录匹配的结果。

    Result of matching a raw ingredient name against a catalogue.

    Attributes:
        canonical_name: The matched catalogue entry's canonical name.
            canonical_name：匹配到的目录条目规范名称。
        catalogue_id: Identifier of the matched catalogue entry (empty if no match).
            catalogue_id：匹配目录条目的标识（无匹配时为空）。
        confidence: Match confidence [0, 1]. 1.0 = exact match, < 0.5 = weak.
            confidence：匹配置信度 [0, 1]。1.0 = 精确匹配，< 0.5 = 弱。
        matched_by: How the match was found: "exact" | "alias" | "partial" | "none".
            matched_by：如何找到匹配："exact" | "alias" | "partial" | "none"。
    """

    canonical_name: str
    catalogue_id: str
    confidence: float
    matched_by: str


# Minimal catalogue entry for MVP — production systems replace this with
# a database-backed ingredient catalogue.
# MVP 的最小目录条目 —— 生产系统用数据库支撑的食材目录替换它。
IngredientCatalogue = dict[str, str]
"""Catalogue mapping: canonical_name → catalogue_id.

目录映射：canonical_name → catalogue_id。

Example:
    {"chicken breast": "cat_poultry_001", "salt": "cat_seasoning_042"}
"""


def match_catalogue_item(
    raw_name: str,
    catalogue: IngredientCatalogue,
) -> CanonicalIngredientMatch:
    """把（可能原始的）食材名与食材目录匹配。

    Match a (possibly raw) ingredient name against an ingredient catalogue.

    Strategy:
      1. Normalise the name via ``normalise_ingredient_name``.
      2. Exact match against catalogue keys (case-insensitive).
      3. Alias-based match: check if any catalogue key is a substring.
      4. Partial match: check token overlap (Jaccard-like).  At least 50 %
         token overlap required.

    策略：
      1. 通过 ``normalise_ingredient_name`` 规范化名称。
      2. 对目录键精确匹配（大小写不敏感）。
      3. 基于别名匹配：检查任意目录键是否为子串。
      4. 部分匹配：检查 token 重叠（类 Jaccard）。至少需 50% token 重叠。

    Args:
        raw_name: Raw ingredient name from recipe text.
            raw_name：来自菜谱文本的原始食材名。
        catalogue: A dict mapping canonical_name → catalogue_id.
            catalogue：映射 canonical_name → catalogue_id 的字典。

    Returns:
        CanonicalIngredientMatch with the best match.  catalogue_id is
        empty and confidence is 0.0 when no match is found.
        带最佳匹配的 CanonicalIngredientMatch。无匹配时 catalogue_id 为空、
        confidence 为 0.0。

    Examples:
        >>> cat = {"chicken breast": "p001", "salt": "s001"}
        >>> match_catalogue_item("Chicken Breast Fillet", cat)
        CanonicalIngredientMatch(canonical_name='chicken breast', catalogue_id='p001', confidence=0.7, matched_by='alias')
        >>> match_catalogue_item("saffron", cat)
        CanonicalIngredientMatch(canonical_name='saffron', catalogue_id='', confidence=0.0, matched_by='none')
    """
    normalised = normalise_ingredient_name(raw_name)

    # --- Step 1: exact match 精确匹配 ---
    if normalised in catalogue:
        return CanonicalIngredientMatch(
            canonical_name=normalised,
            catalogue_id=catalogue[normalised],
            confidence=1.0,
            matched_by="exact",
        )

    # --- Step 2: normalised-name-as-alias lookup 规范化名作为别名查找 ---
    # Try matching the normalised name against catalogue keys via alias mapping.
    # This catches cases where normalise_ingredient_name didn't resolve to a
    # catalogue key but the catalogue key is itself a broader category.
    # 尝试通过别名映射把规范化名与目录键匹配。这能捕获
    # normalise_ingredient_name 未解析到目录键、但目录键本身是更宽泛类别的情况。
    for cat_key, cat_id in catalogue.items():
        if normalised == cat_key:
            return CanonicalIngredientMatch(
                canonical_name=cat_key,
                catalogue_id=cat_id,
                confidence=1.0,
                matched_by="exact",
            )

    # --- Step 3: substring / alias match 子串 / 别名匹配 ---
    # Check if any catalogue key appears as a substring of the normalised name.
    # 检查任意目录键是否作为规范化名的子串出现。
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

    # --- Step 4: token-overlap partial match token 重叠部分匹配 ---
    # Split both sides into tokens.  If >= 50 % of the raw name's tokens
    # appear in a catalogue key (or vice versa), accept as partial match.
    # 把两侧拆成 token。若 >= 50% 的原始名 token 出现在目录键中（或反之），接受为部分匹配。
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

    # --- No match 无匹配 ---
    return CanonicalIngredientMatch(
        canonical_name=normalised,
        catalogue_id="",
        confidence=0.0,
        matched_by="none",
    )
