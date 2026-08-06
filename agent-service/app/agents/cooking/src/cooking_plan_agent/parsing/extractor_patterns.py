"""Static patterns for the rule-based recipe extractor — parses preprocessed text into structured candidates.

Handbook 4.3–4.8: this module implements the RecipeExtractor Protocol
defined in workflow/context.py. In MVP, extraction is rule-based (regex +
keyword matching). When an LLM adapter is wired later, this module is
replaced while keeping the Protocol contract unchanged.

Architecture (Ports & Adapters):
  - Port:  RecipeExtractor Protocol (workflow/context.py)
  - Adapter (this file): rule-based implementation
  - Future adapter: LLM-based with structured output

Supported patterns:
  - Chinese recipes: "食材/原料" section, "步骤/做法" section
  - English recipes: "Ingredients" section, "Steps/Directions" section
  - Mixed-language recipes detected by preprocess pipeline
"""

from __future__ import annotations

import re

from cooking_plan_agent.domain.enums import HeatLevel

# =============================================================================
# Ingredient line patterns
# =============================================================================

# Western: "200g chicken breast, diced" or "1 tbsp soy sauce"
# Group 1: quantity  Group 2: unit  Group 3: name  Group 4: preparation
_RE_INGREDIENT_WESTERN = re.compile(
    r"^[-*•\s]*"  # Leading list marker
    r"(\d+(?:\.\d+)?)\s*"  # Quantity
    r"(g|kg|mg|ml|l|cl|dl|tbsp|tsp|cups?|cup|oz|lb|lbs?|piece|pcs?|pc)?\s+"  # Optional unit
    r"(.+?)$",  # Name + optional prep
    re.IGNORECASE,
)

# Chinese: "鸡胸肉 200g，切丁" or "番茄 2个" or "大蒜3-4瓣"
# Group 1: name  Group 2: quantity (lo)  Group 3 (optional): range high
# Group 4 (optional): unit  Group 5 (optional): preparation
_RE_INGREDIENT_CHINESE = re.compile(
    r"^[-*•、\s]*"
    r"([\u4e00-\u9fff\w]+?)\s*"  # Name (Chinese characters or word chars)
    r"(\d+(?:\.\d+)?)\s*"  # Quantity (low end)
    r"(?:[-~—–至到]\s*(\d+(?:\.\d+)?)\s*)?"  # Optional quantity range high (e.g. "3-4")
    r"(克|g|kg|mg|毫升|ml|l|升|个|根|颗|只|条|块|片|把|瓣|勺|汤匙|茶匙|杯|碗|两|斤)?\s*"  # Optional unit (Chinese or Latin)
    r"[，,]?\s*(.+)?$",  # Optional preparation note
)

# Chinese unit → standard unit mapping
_CHINESE_UNIT_MAP: dict[str, str] = {
    "克": "g",
    "毫升": "ml",
    "升": "l",
    "个": "piece",
    "根": "piece",
    "颗": "piece",
    "只": "piece",
    "条": "piece",
    "块": "piece",
    "片": "piece",
    "瓣": "piece",
    "把": "piece",
    "勺": "tbsp",
    "汤匙": "tbsp",
    "茶匙": "tsp",
    "杯": "cup",
    "碗": "cup",
    "两": "liang",
    "斤": "jin",
}

# Ingredients with no quantity — "to taste", "适量", "少许", "老抽少许"
_RE_NO_QUANTITY = re.compile(
    r"^(适量|少许|若干|to\s+taste|a\s+pinch|a\s+dash|salt\s+and\s+pepper)",
    re.IGNORECASE,
)

# Quantity qualifier appearing anywhere in a line ("适量", "少许", "少量",
# "若干", "to taste", "a pinch", "a dash") — signals a no-quantity ingredient
# even when the qualifier trails the name (e.g. "老抽少许").
_RE_QUANTITY_QUALIFIER = re.compile(
    r"(适量|少许|少量|若干|to\s+taste|a\s+pinch|a\s+dash)",
    re.IGNORECASE,
)

# Ingredient-name noise to strip during cleaning (extractor stage):
#   - parenthetical notes: "味精/鸡精（可选）" → "味精/鸡精"; "小米辣（依吃辣程度放）" → "小米辣"
#   - trailing quantity qualifiers: "老抽少许" → "老抽"
#   - trailing punctuation: "白胡椒粉、" → "白胡椒粉"
_RE_PAREN_NOTE = re.compile(r"[（(][^）)]*[）)]")
_RE_TRAILING_QUANTITY = re.compile(r"(?:少许|适量|少量|若干)$")
_RE_TRAILING_PUNCT = re.compile(r"[、。，,;；·]+$")

# Ingredient name + preparation note splitter
# "chicken breast, diced" → name="chicken breast", prep="diced"
# "鸡蛋，打散" → name="鸡蛋", prep="打散"
_RE_NAME_PREP_SPLIT = re.compile(r"\s*[，,]\s*(.+)$")

# Preparation keywords for detection
_PREP_KEYWORDS = frozenset(
    {
        "diced",
        "dice",
        "minced",
        "mince",
        "chopped",
        "chop",
        "sliced",
        "slice",
        "julienned",
        "julienne",
        "grated",
        "grate",
        "crushed",
        "crush",
        "peeled",
        "peel",
        "washed",
        "wash",
        "切丁",
        "切块",
        "切片",
        "切丝",
        "剁碎",
        "切末",
        "打散",
        "去皮",
        "洗净",
        "泡发",
        "切段",
        "切圈",
    }
)


# =============================================================================
# Step section detection
# =============================================================================

# Section headers that indicate ingredient/step boundaries
_STEP_HEADER_PATTERNS = [
    re.compile(
        r"^(?:steps?|directions?|instructions?|method|做法|步骤|制作方法|制作步骤|烹饪步骤)\s*[：:]*\s*$", re.IGNORECASE
    ),
]
_INGREDIENT_HEADER_PATTERNS = [
    re.compile(
        r"^(?:ingredients?|what\s+you(?:'ll)?\s+need|食材|食材准备|原料|配料|用料|材料|主料|辅料|调料)\s*[：:]*\s*$",
        re.IGNORECASE,
    ),
]

# Step number patterns
_RE_STEP_NUMBER = re.compile(
    r"^(?:\d+[.\)、]\s*)"  # "1.", "2)", "3、"
    r"|^(?:step\s*\d+|步骤\s*\d+)\s*[：:.]?\s*",  # "Step 1", "步骤1"
    re.IGNORECASE,
)

# =============================================================================
# Cooking technique detection (step analysis)
# =============================================================================

# Technique → pattern mapping. Ordered: check longer/compound names first.
_TECHNIQUE_PATTERNS: list[tuple[str, str, str]] = [
    # (technique, english_pattern, chinese_pattern)
    # Ordered: check longer/compound names first, then general patterns
    ("stir_fry", r"\bstir[-\s]?fr(?:y|ied|ying)\b", r"炒|爆炒|翻炒"),
    ("deep_fry", r"\bdeep[-\s]?fr(?:y|ied|ying)\b", r"炸|油炸"),
    ("boil", r"\bboil(?:ing|ed)?\b", r"煮|烧开|煮沸|焯"),
    ("simmer", r"\bsimmer(?:ing|ed)?\b", r"焖|炖|煲|慢炖|小火炖"),
    ("steam", r"\bsteam(?:ing|ed)?\b", r"蒸"),
    ("bake", r"\bbak(?:e|ing|ed)\b", r"烤|烘烤"),
    ("roast", r"\broast(?:ing|ed)?\b", r"烤|烘"),
    ("grill", r"\bgrill(?:ing|ed)?\b", r"煎|烧烤"),
    ("marinate", r"\bmarinat(?:e|ing|ed)\b", r"腌|腌制"),
    ("sauté", r"\bsauté(?:ing|ed)?\b", r"煎|煸"),
    ("sear", r"\bsear(?:ing|ed)?\b", r"煎|封"),
    ("braise", r"\bbrais(?:e|ing|ed)\b", r"红烧|卤"),
    ("poach", r"\bpoach(?:ing|ed)?\b", r"水煮|清煮"),
    # General heating — match "heat" in cooking context (not weather)
    ("heat", r"\bheat(?:ing|ed)?\s+(?:oil|pan|wok|pot|oven)\b", r"热锅|烧热|加热"),
]

# Heat level detection
_HEAT_LEVELS: list[tuple[str, HeatLevel]] = [
    (r"high\s+(?:heat|flame|temperature)", HeatLevel.HIGH),
    (r"大火|猛火|旺火|高火", HeatLevel.HIGH),
    (r"medium[-\s]high\s+(?:heat|flame)", HeatLevel.HIGH),
    (r"中大火|中高火", HeatLevel.HIGH),
    (r"medium(?![\s-](?:high|low))\s+(?:heat|flame)", HeatLevel.MEDIUM),
    (r"中火(?!\s*[大旺猛])", HeatLevel.MEDIUM),
    (r"medium[-\s]low\s+(?:heat|flame)", HeatLevel.LOW),
    (r"low\s+(?:heat|flame|temperature)", HeatLevel.LOW),
    (r"小火|文火|微火|慢火", HeatLevel.LOW),
]

# Duration patterns
_RE_DURATION_RANGE = re.compile(
    r"(\d+)\s*[-–—~to]+\s*(\d+)\s*(?:分钟|min(?:ute)?s?)",
    re.IGNORECASE,
)
_RE_DURATION_SINGLE = re.compile(
    r"(?:about|approximately|around|大约|约|大概|腌制)?\s*(\d+)\s*(?:分钟|min(?:ute)?s?)\b",
    re.IGNORECASE,
)

# Resource hints
_RESOURCE_KEYWORDS: dict[str, str] = {
    "oven": r"\boven\b|烤箱|烤炉",
    "stove": r"\bstove\b|炉灶|灶台|炉子",
    "wok": r"\bwok\b|炒锅|铁锅",
    "pan": r"\b(?:frying\s+)?pan\b|平底锅",
    "pot": r"\bpot\b|锅|汤锅",
    "steamer": r"\bsteamer\b|蒸锅|蒸笼",
    "sink": r"\bsink\b|水槽|水池",
    "cutting_board": r"\bcutting\s+board\b|砧板|菜板|案板",
    "knife": r"\bknife\b|刀",
    "mixing_bowl": r"\bmixing\s+bowl\b|搅拌碗|大碗",
}


# =============================================================================
# RecipeExtractor — rule-based implementation
# =============================================================================
