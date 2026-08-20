# =============================================================================
# 规则提取器的静态模式模块（parsing/extractor_patterns）
# -----------------------------------------------------------------------------
# 定义“基于规则的菜谱提取器”所用的静态正则 / 关键词模式，把预处理后的文本
# 解析为结构化候选。实现手册 4.3–4.8。
# 架构（端口与适配器）：
#   - Port    ：RecipeExtractor 协议（workflow/context.py）
#   - Adapter ：本文件的基于规则实现（regex + 关键词匹配）
#   - 未来适配器：基于 LLM 的结构化输出（保持协议不变，仅替换本模块）
# 支持的模式：
#   - 中文菜谱："食材/原料" 段、"步骤/做法" 段
#   - 英文菜谱："Ingredients" 段、"Steps/Directions" 段
#   - 混合语言菜谱（由预处理流水线检测）
# =============================================================================

"""Static patterns for the rule-based recipe extractor — parses preprocessed text into structured candidates.

规则提取器的静态模式 —— 把预处理文本解析为结构化候选。

Handbook 4.3–4.8: this module implements the RecipeExtractor Protocol
defined in workflow/context.py. In MVP, extraction is rule-based (regex +
keyword matching). When an LLM adapter is wired later, this module is
replaced while keeping the Protocol contract unchanged.

手册 4.3–4.8：本模块实现 workflow/context.py 中定义的 RecipeExtractor 协议。
MVP 阶段提取是规则式的（regex + 关键词匹配）。后续接入 LLM 适配器时，
本模块被替换，但协议契约保持不变。

Architecture (Ports & Adapters):
  - Port:  RecipeExtractor Protocol (workflow/context.py)
  - Adapter (this file): rule-based implementation
  - Future adapter: LLM-based with structured output

架构（端口与适配器）：
  - Port：RecipeExtractor 协议（workflow/context.py）
  - Adapter（本文件）：基于规则实现
  - 未来适配器：基于 LLM 的结构化输出

Supported patterns:
  - Chinese recipes: "食材/原料" section, "步骤/做法" section
  - English recipes: "Ingredients" section, "Steps/Directions" section
  - Mixed-language recipes detected by preprocess pipeline

支持的模式：
  - 中文菜谱："食材/原料" 段、"步骤/做法" 段
  - 英文菜谱："Ingredients" 段、"Steps/Directions" 段
  - 混合语言菜谱（由预处理流水线检测）
"""

from __future__ import annotations

import re

from cooking_plan_agent.domain.enums import HeatLevel

# =============================================================================
# Ingredient line patterns
# 食材行模式
# =============================================================================

# Western: "200g chicken breast, diced" or "1 tbsp soy sauce"
# Group 1: quantity  Group 2: unit  Group 3: name  Group 4: preparation
# 西式："200g chicken breast, diced" 或 "1 tbsp soy sauce"
# 分组1：数量  分组2：单位  分组3：名称  分组4：预处理
_RE_INGREDIENT_WESTERN = re.compile(
    r"^[-*•\s]*"  # Leading list marker  前导列表标记
    r"(\d+(?:\.\d+)?)\s*"  # Quantity  数量
    r"(grams?|kilograms?|milligrams?|millilit(?:er|re)s?|lit(?:er|re)s?|"
    r"tablespoons?|teaspoons?|ounces?|pounds?|pieces?|"
    r"g|kg|mg|ml|l|cl|dl|tbsp|tsp|cups?|oz|lb|lbs?|pcs?|pc)?\s+"  # Optional unit  可选单位
    r"(.+?)$",  # Name + optional prep  名称 + 可选预处理
    re.IGNORECASE,
)

# Quantity-first CJK: "400克硬豆腐", "3个鸡蛋", "20 克 大蒜".  This is
# deliberately separate from the name-first Chinese pattern below so a
# leading quantity can never be mistaken for the ingredient name.
# 数量前置的中文："400克硬豆腐"、"3个鸡蛋"、"20 克 大蒜"。
# 刻意与下方的“名称前置中文模式”分开，使前导数量绝不会被误认为食材名。
_RE_INGREDIENT_CJK_QUANTITY_FIRST = re.compile(
    r"^[-*•、\s]*"
    r"(\d+(?:\.\d+)?)\s*"
    r"(克|千克|公斤|毫克|毫升|升|个|根|颗|只|条|块|片|把|瓣|勺|汤匙|茶匙|杯|碗|两|斤)\s*"
    r"(.+?)\s*$",
    re.IGNORECASE,
)

# Chinese: "鸡胸肉 200g，切丁" or "番茄 2个" or "大蒜3-4瓣"
# Group 1: name  Group 2: quantity (lo)  Group 3 (optional): range high
# Group 4 (optional): unit  Group 5 (optional): preparation
# 中文："鸡胸肉 200g，切丁" 或 "番茄 2个" 或 "大蒜3-4瓣"
# 分组1：名称  分组2：数量（下限）  分组3（可选）：区间上限
# 分组4（可选）：单位  分组5（可选）：预处理
_RE_INGREDIENT_CHINESE = re.compile(
    r"^[-*•、\s]*"
    r"([\u4e00-\u9fff\w]+?)\s*"  # Name (Chinese characters or word chars)  名称（中文字符或单词字符）
    r"(\d+(?:\.\d+)?)\s*"  # Quantity (low end)  数量（下限）
    r"(?:[-~—–至到]\s*(\d+(?:\.\d+)?)\s*)?"  # Optional quantity range high (e.g. "3-4")  可选数量区间上限（如 "3-4"）
    r"(克|g|kg|mg|毫升|ml|l|升|个|根|颗|只|条|块|片|把|瓣|勺|汤匙|茶匙|杯|碗|两|斤)?\s*"  # Optional unit (Chinese or Latin)  可选单位（中文或拉丁）
    r"[，,]?\s*(.+)?$",  # Optional preparation note  可选预处理备注
)

# Chinese unit → standard unit mapping
# 中文单位 → 标准单位映射
_CHINESE_UNIT_MAP: dict[str, str] = {
    "克": "g",
    "千克": "kg",
    "公斤": "kg",
    "毫克": "mg",
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
# 无数量食材 —— "to taste"、"适量"、"少许"、"老抽少许"
_RE_NO_QUANTITY = re.compile(
    r"^(适量|少许|若干|to\s+taste|a\s+pinch|a\s+dash|salt\s+and\s+pepper)",
    re.IGNORECASE,
)

# Quantity qualifier appearing anywhere in a line ("适量", "少许", "少量",
# "若干", "to taste", "a pinch", "a dash") — signals a no-quantity ingredient
# even when the qualifier trails the name (e.g. "老抽少许").
# 出现在行内任意位置的数量限定词（"适量"、"少许"、"少量"、"若干"、"to taste"、
# "a pinch"、"a dash"）—— 即使限定词跟在名称后（如 "老抽少许"），也标记为无数量食材。
_RE_QUANTITY_QUALIFIER = re.compile(
    r"(适量|少许|少量|若干|to\s+taste|a\s+pinch|a\s+dash)",
    re.IGNORECASE,
)

# Ingredient-name noise to strip during cleaning (extractor stage):
#   - parenthetical notes: "味精/鸡精（可选）" → "味精/鸡精"; "小米辣（依吃辣程度放）" → "小米辣"
#   - trailing quantity qualifiers: "老抽少许" → "老抽"
#   - trailing punctuation: "白胡椒粉、" → "白胡椒粉"
# 清洗阶段（提取器阶段）需剥离的食材名噪声：
#   - 括号备注："味精/鸡精（可选）" → "味精/鸡精"；"小米辣（依吃辣程度放）" → "小米辣"
#   - 尾随数量限定词："老抽少许" → "老抽"
#   - 尾随标点："白胡椒粉、" → "白胡椒粉"
_RE_PAREN_NOTE = re.compile(r"[（(][^）)]*[）)]")
_RE_TRAILING_QUANTITY = re.compile(r"(?:少许|适量|少量|若干)$")
_RE_TRAILING_PUNCT = re.compile(r"[、。，,;；·]+$")

# Ingredient name + preparation note splitter
# "chicken breast, diced" → name="chicken breast", prep="diced"
# "鸡蛋，打散" → name="鸡蛋", prep="打散"
# 食材名 + 预处理备注切分器
_RE_NAME_PREP_SPLIT = re.compile(r"\s*[，,]\s*(.+)$")

# Preparation keywords for detection
# 用于检测的预处理关键词
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
# 步骤段检测
# =============================================================================

# Section headers that indicate ingredient/step boundaries
# 指示食材 / 步骤边界的段标题
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
# 步骤编号模式
_RE_STEP_NUMBER = re.compile(
    r"^(?:\d+[.\)、]\s*)"  # "1.", "2)", "3、"
    r"|^(?:step\s*\d+|步骤\s*\d+)\s*[：:.]?\s*",  # "Step 1", "步骤1"
    re.IGNORECASE,
)

# =============================================================================
# Cooking technique detection (step analysis)
# 烹饪技法检测（步骤分析）
# =============================================================================

# Technique → pattern mapping. Ordered: check longer/compound names first.
# 技法 → 模式映射。有序：先检查较长 / 复合名称。
_TECHNIQUE_PATTERNS: list[tuple[str, str, str]] = [
    # (technique, english_pattern, chinese_pattern)
    # (技法, 英文模式, 中文模式)
    # Ordered: check longer/compound names first, then general patterns
    # 有序：先检查较长 / 复合名称，再检查通用模式
    ("stir_fry", r"\bstir[-\s]?fr(?:y|ied|ying)\b", r"炒|爆炒|翻炒"),
    ("deep_fry", r"\bdeep[-\s]?fr(?:y|ied|ying)\b", r"炸|油炸"),
    ("boil", r"\bboil(?:ing|ed)?\b", r"煮|烧开|煮沸|焯"),
    ("simmer", r"\bsimmer(?:ing|ed)?\b", r"焖|炖|煲|慢炖|小火炖"),
    ("steam", r"\bsteam(?:ing|ed)?\b", r"蒸"),
    ("bake", r"\bbak(?:e|ing|ed)\b", r"烤|烘烤"),
    ("roast", r"\broast(?:ing|ed)?\b", r"烤|烘"),
    ("grill", r"\bgrill(?:ing|ed)?\b", r"煎|烧烤"),
    ("marinate", r"\bmarinat(?:e|ing|ed)\b", r"腌|腌制"),
    (
        "sauté",
        r"\b(?:pan[-\s]?fr(?:y|ied|ying)|saut(?:é|e)(?:ing|ed)?)\b",
        r"煎|煸",
    ),
    ("sear", r"\bsear(?:ing|ed)?\b", r"煎|封"),
    ("braise", r"\bbrais(?:e|ing|ed)\b", r"红烧|卤"),
    ("poach", r"\bpoach(?:ing|ed)?\b", r"水煮|清煮"),
    # General heating — match "heat" in cooking context (not weather)
    # 通用加热 —— 匹配烹饪语境下的 "heat"（而非天气）
    ("heat", r"\bheat(?:ing|ed)?\s+(?:oil|pan|wok|pot|oven)\b", r"热锅|烧热|加热"),
]

# Heat level detection
# 火力档位检测
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
# 时长模式
_RE_DURATION_RANGE = re.compile(
    r"(\d+)\s*[-–—~to]+\s*(\d+)\s*(?:分钟|min(?:ute)?s?)",
    re.IGNORECASE,
)
_RE_DURATION_SINGLE = re.compile(
    r"(?:about|approximately|around|大约|约|大概|腌制)?\s*(\d+)\s*(?:分钟|min(?:ute)?s?)\b",
    re.IGNORECASE,
)

# Resource hints
# 资源提示
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
# 菜谱提取器 —— 基于规则实现
# =============================================================================
