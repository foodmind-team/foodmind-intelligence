# =============================================================================
# 基于规则的菜谱提取适配器模块（parsing/extractor）
# -----------------------------------------------------------------------------
# 实现 RecipeExtractor 协议，把预处理后的菜谱文本解析为结构化候选。
# 解析正则模式集中在 extractor_patterns 中，本模块专注“文本 → 领域候选”的转换。
# 无 LLM 依赖 —— MVP 阶段用 regex + 关键词匹配。
# 核心职责：
#   - _split_sections      ：把行分成“食材段”与“步骤段”
#   - _parse_ingredient    ：解析单行食材（西式 / 中文数量前置 / 中文名称前置 / 无数量）
#   - _parse_step          ：解析单行步骤（技法 / 火力 / 时长 / 温度 / 资源）
#   - _extract_dish_name   ：提取菜名（容忍网页样板文）
# =============================================================================

"""Rule-based recipe extraction adapter.

基于规则的菜谱提取适配器。

Parsing patterns live in ``extractor_patterns`` so this module focuses on
turning text into domain candidates.

解析模式集中在 extractor_patterns 中，因此本模块专注把文本转换为领域候选。
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
    _RE_INGREDIENT_CJK_QUANTITY_FIRST,
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

# 菜谱元数据段标题（注释 / 营养信息 / 烹饪模式 / 小贴士），用于在步骤段中截断
_RECIPE_METADATA_HEADER = re.compile(
    r"^\s*(?:recipe\s+notes?|notes?|nutrition(?:al)?(?:\s+information)?|cook\s+mode|tips?)\s*:?[\s]*$",
    re.IGNORECASE,
)


class RecipeExtractor:
    """基于规则的菜谱文本提取器，实现 RecipeExtractor 协议。

    Rule-based recipe text extractor implementing the RecipeExtractor Protocol.

    Extracts structured ExtractedRecipeCandidate from preprocessed recipe text.
    No LLM dependency — uses regex and keyword matching for MVP.

    从预处理后的菜谱文本提取结构化 ExtractedRecipeCandidate。
    无 LLM 依赖 —— MVP 阶段用 regex 与关键词匹配。

    Protocol contract (from workflow/context.py):
        async def extract(self, source_text: str) -> ExtractedRecipeCandidate
    """

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        """把预处理后的菜谱文本解析为结构化候选。

        Parse preprocessed recipe text into a structured candidate.

        Args:
            source_text: Preprocessed recipe text (decoded, normalized, cleaned).
                source_text：预处理后的菜谱文本（已解码、规范化、清洗）。

        Returns:
            ExtractedRecipeCandidate with ingredients and steps.
            含食材与步骤的 ExtractedRecipeCandidate。
        """
        lines = source_text.strip().split("\n")

        # Separate ingredient and step sections
        # 分离食材段与步骤段
        ingredient_lines, step_lines = self._split_sections(lines)

        # Parse each section; expand "味精/鸡精" style slash alternatives
        # into separate ingredient candidates so inventory matching works.
        # 解析每个段；把 "味精/鸡精" 这类斜杠可选项展开为独立食材候选，使库存匹配可用。
        parsed_ingredients = tuple(self._parse_ingredient(line) for line in ingredient_lines)
        ingredients = tuple(
            sub for ing in parsed_ingredients if ing is not None for sub in self._expand_slash_alternatives(ing)
        )

        steps = tuple(self._parse_step(i + 1, line) for i, line in enumerate(step_lines))

        # Generate a stable recipe_id from the first line (dish name)
        # 从首行（菜名）生成稳定的 recipe_id
        dish_name = self._extract_dish_name(lines)
        recipe_id = self._make_recipe_id(dish_name)

        # Detect language from the text content
        # 从文本内容检测语言
        source_language = self._detect_language(source_text)

        # Default to 2 servings when not specified
        # 未指定份数时默认 2 份
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
    # 段切分
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sections(lines: list[str]) -> tuple[list[str], list[str]]:
        """把菜谱行切成食材段与步骤段。

        Split recipe lines into ingredient and step sections.

        Detects section headers ("Ingredients:", "Steps:", "食材:", "步骤:") to
        determine boundaries. Falls back to heuristics if no headers found.

        检测段标题（"Ingredients:"、"Steps:"、"食材:"、"步骤:"）确定边界。
        若无标题则回退到启发式。
        """
        ingredient_start: int | None = None
        step_start: int | None = None

        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            # Check ingredient headers  检查食材标题
            if ingredient_start is None:
                for pat in _INGREDIENT_HEADER_PATTERNS:
                    if pat.match(stripped):
                        ingredient_start = i
                        break
            # Check step headers  检查步骤标题
            if step_start is None:
                for pat in _STEP_HEADER_PATTERNS:
                    if pat.match(stripped):
                        step_start = i
                        break

        if ingredient_start is not None and step_start is not None:
            # Both sections found  两段都找到
            ing_lines = lines[ingredient_start + 1 : step_start]
            step_end = next(
                (index for index in range(step_start + 1, len(lines)) if _RECIPE_METADATA_HEADER.match(lines[index])),
                len(lines),
            )
            step_lines = lines[step_start + 1 : step_end]
        elif step_start is not None:
            # Only step section found — everything before is ingredients
            # 只找到步骤段 —— 之前全部视为食材
            ing_lines = lines[:step_start]
            step_lines = lines[step_start + 1 :]
        else:
            # No explicit sections — use heuristics
            # 无显式段 —— 使用启发式
            ing_lines, step_lines = RecipeExtractor._heuristic_split(lines)

        # Filter empty lines and strip
        # 过滤空行并去空白
        ing_lines = [line.strip() for line in ing_lines if line.strip()]
        step_lines = [line.strip() for line in step_lines if line.strip()]

        return ing_lines, step_lines

    @staticmethod
    def _heuristic_split(lines: list[str]) -> tuple[list[str], list[str]]:
        """启发式切分：带编号的行是步骤，其余是食材。"""
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
        # 若未找到编号步骤，把所有行当作一个整体块
        if not step_lines:
            step_lines = ing_lines
            ing_lines = []

        return ing_lines, step_lines

    # ------------------------------------------------------------------
    # Ingredient parsing
    # 食材解析
    # ------------------------------------------------------------------

    def _parse_ingredient(self, line: str) -> ExtractedIngredient | None:
        """把单行食材解析为 ExtractedIngredient。

        Parse a single ingredient line into ExtractedIngredient.

        Returns None if the line cannot be parsed as an ingredient.

        若该行无法解析为食材则返回 None。
        """
        if not line.strip():
            return None

        # Skip section headers  跳过段标题
        stripped_lower = line.strip().lower()
        for pat in _INGREDIENT_HEADER_PATTERNS:
            if pat.match(stripped_lower):
                return None
        for pat in _STEP_HEADER_PATTERNS:
            if pat.match(stripped_lower):
                return None

        # Try Western pattern first, then Chinese
        # 先试西式，再试中文
        result = (
            self._try_cjk_quantity_first(line)
            or self._try_western(line)
            or self._try_chinese(line)
            or self._try_no_quantity(line)
        )
        return result

    def _try_cjk_quantity_first(self, line: str) -> ExtractedIngredient | None:
        """解析紧凑的数量前置中文食材，如 ``400克豆腐``。"""
        match = _RE_INGREDIENT_CJK_QUANTITY_FIRST.match(line.strip())
        if not match:
            return None
        quantity, unit_raw, rest = match.groups()
        name, prep = self._split_name_prep(rest)
        name = self._clean_ingredient_name(name)
        if not name:
            return None
        return ExtractedIngredient(
            raw_text=line.strip(),
            name=name,
            quantity=Decimal(quantity),
            unit=_normalise_unit(unit_raw),
            preparation=prep.strip() if prep else None,
            extraction_source="EXPLICIT",
            confidence=Decimal("0.9"),
        )

    def _try_western(self, line: str) -> ExtractedIngredient | None:
        """尝试西式食材模式：'200g chicken breast, diced'。"""
        match = _RE_INGREDIENT_WESTERN.match(line.strip())
        if not match:
            return None

        quantity_str = match.group(1)
        unit = (match.group(2) or "").lower()
        rest = match.group(3).strip()

        # Split name and preparation  切分名称与预处理
        name, prep = self._split_name_prep(rest)

        # Validate: name should be non-empty and look like a food item
        # 校验：名称应非空且看起来像食物
        if not name or len(name) < 2:
            return None

        # Map unit to canonical form  把单位映射到规范形式
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
        """尝试中文食材模式：'鸡胸肉 200g，切丁' 或 '大蒜3-4瓣'。"""
        match = _RE_INGREDIENT_CHINESE.match(line.strip())
        if not match:
            return None

        name = match.group(1).strip()
        quantity_lo = match.group(2)
        quantity_hi = match.group(3)
        unit_raw = match.group(4)
        prep_raw = match.group(5)

        # Validate name  校验名称
        if not name or len(name) < 1:
            return None

        # Map Chinese units  映射中文单位
        unit = _normalise_unit(unit_raw.strip()) if unit_raw else "piece"

        # Quantity ranges ("3-4瓣") take the upper bound — conservative so the
        # plan never under-supplies (mirrors serving-range handling).
        # 数量区间（"3-4瓣"）取上限 —— 保守策略，使计划绝不供应不足（与份数区间处理一致）。
        quantity = Decimal(quantity_hi) if quantity_hi else Decimal(quantity_lo)

        # Clean preparation  清洗预处理
        prep = prep_raw.strip() if prep_raw else None
        if prep:
            prep = _RE_STEP_NUMBER.sub("", prep).strip()  # Remove stray step numbers  移除混入的步骤编号

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
        """处理无数量食材：'salt to taste'、'适量盐'、'老抽少许'。"""
        stripped = line.strip()
        if not stripped:
            return None

        # Clean name noise BEFORE classification: parenthetical notes
        # ("味精/鸡精（可选）", "小米辣（依吃辣程度放）"), trailing qualifiers
        # ("老抽少许"), and trailing punctuation ("白胡椒粉、").
        # 在分类之前先清洗名称噪声：括号备注（"味精/鸡精（可选）"、"小米辣（依吃辣程度放）"）、
        # 尾随限定词（"老抽少许"）、尾随标点（"白胡椒粉、"）。
        cleaned = self._clean_ingredient_name(stripped)
        if not cleaned:
            return None

        # Any quantity qualifier anywhere in the line ("适量", "少许", ...)
        # marks this as a no-quantity ingredient line.
        # 行内任意位置出现数量限定词（"适量"、"少许" 等）都标记为无数量食材行。
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
        # 短文本（可能是裸食材名）且不像步骤指令 → 自由文本食材。
        # 分类在清洗后的名称上运行，因此括号备注不会触发步骤指示词。
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
        """把 "味精/鸡精" 这类可选项展开为独立食材。

        Expand "味精/鸡精" style alternatives into separate ingredients.

        A slash in a no-quantity Chinese ingredient line usually means "or"
        ("味精/鸡精" = MSG or chicken powder). Splitting into one demand per
        alternative lets each match inventory independently; a lone slash with
        no CJK either side (e.g. "A/B sauce") is left untouched.

        无数量中文食材行中的斜杠通常表示“或”（"味精/鸡精" = 味精或鸡粉）。
        拆成每个可选项一条需求，使各自能独立匹配库存；斜杠两侧无中文（如 "A/B sauce"）则不动。
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
        """剥离括号备注、尾随限定词与尾随标点。

        Strip parenthetical notes, trailing qualifiers, and trailing punctuation.

        Applied to no-quantity ingredient lines so downstream inventory
        matching sees clean canonical names:
          "味精/鸡精（可选）"  → "味精/鸡精"
          "小米辣（依吃辣程度放）" → "小米辣"
          "老抽少许"           → "老抽"
          "白胡椒粉、"          → "白胡椒粉"

        应用于无数量食材行，使下游库存匹配看到干净的规范名称。
        """
        name = _RE_PAREN_NOTE.sub("", text).strip()
        name = _RE_TRAILING_QUANTITY.sub("", name).strip()
        name = _RE_TRAILING_PUNCT.sub("", name).strip()
        return name

    @staticmethod
    def _split_name_prep(text: str) -> tuple[str, str | None]:
        """把 'chicken breast, diced' 切分为 (name, prep)。"""
        match = _RE_NAME_PREP_SPLIT.search(text)
        if not match:
            return text.strip(), None

        name_part = text[: match.start()].strip()
        prep_part = match.group(1).strip()

        # Only treat as preparation if it contains a known prep keyword
        # 仅当包含已知预处理关键词时才视为预处理
        prep_lower = prep_part.lower()
        if any(kw in prep_lower for kw in _PREP_KEYWORDS):
            return name_part, prep_part

        # If the "prep" part is very short and doesn't look like prep,
        # it might be part of the name
        # 若“prep”部分非常短且不像预处理，它可能是名称的一部分
        return text.strip(), None

    # ------------------------------------------------------------------
    # Step parsing
    # 步骤解析
    # ------------------------------------------------------------------

    def _parse_step(self, index: int, line: str) -> ExtractedStep:
        """把步骤行解析为 ExtractedStep。"""
        # Remove step number prefix for cleaner instruction text
        # 移除步骤编号前缀，使指令文本更干净
        instruction = _RE_STEP_NUMBER.sub("", line).strip()

        # Detect cooking technique  检测烹饪技法
        technique = self._detect_technique(line)

        # Map technique to category  把技法映射为类别
        category = self._infer_category(technique)

        # Detect heat level  检测火力档位
        heat = self._detect_heat(line)

        # Detect durations (active and passive)  检测时长（主动与被动）
        active_dur, passive_dur = self._detect_durations(line, technique)

        # Detect temperature  检测温度
        temp = self._detect_temperature(line)

        # Detect resource hints  检测资源提示
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
        """从步骤文本检测主要烹饪技法。"""
        text_lower = text.lower()
        for technique, en_pat, zh_pat in _TECHNIQUE_PATTERNS:
            if re.search(en_pat, text_lower) or re.search(zh_pat, text):
                return technique
        return "general"

    @staticmethod
    def _infer_category(technique: str) -> str:
        """把技法映射为步骤类别。"""
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
        """从步骤文本检测火力档位。"""
        text_lower = text.lower()
        for pat, level in _HEAT_LEVELS:
            if re.search(pat, text_lower):
                return level
        return HeatLevel.NONE

    @staticmethod
    def _detect_durations(text: str, technique: str) -> tuple[int | None, int | None]:
        """从步骤文本提取主动与被动时长。"""
        # Try range: "10-15 minutes", "3~5分钟"  尝试区间
        match = _RE_DURATION_RANGE.search(text)
        if match:
            _lo, hi = int(match.group(1)), int(match.group(2))
            # For passive techniques (boil, simmer, bake, roast), duration is passive
            # 对被动技法（煮、炖、烤、烘焙），时长是被动时长
            passive_techniques = {"boil", "simmer", "bake", "roast", "marinate", "steam", "braise"}
            if technique in passive_techniques:
                return None, hi  # Range max is passive, no explicit active  区间最大值是被动，无显式主动
            return hi, None  # Active technique: treat as active duration  主动技法：视为主动时长

        # Try single: "10 minutes", "5分钟"  尝试单个时长
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
        """从步骤文本检测目标温度。"""
        # Celsius: "180°C", "180 C", "200度"  摄氏度
        match = re.search(r"(\d{2,3})\s*(?:°\s*)?[cC](?:elsius)?\b", text)
        if match:
            return Decimal(match.group(1))

        match = re.search(r"(\d{2,3})\s*度\b", text)
        if match:
            return Decimal(match.group(1))

        # Fahrenheit: "350°F" → Celsius  华氏度 → 摄氏度
        match = re.search(r"(\d{2,4})\s*(?:°\s*)?[fF](?:ahrenheit)?\b", text)
        if match:
            f = Decimal(match.group(1))
            return ((f - 32) * 5 / 9).quantize(Decimal("0.1"))

        return None

    @staticmethod
    def _detect_resources(text: str) -> list[str]:
        """从步骤文本检测所需厨房资源。"""
        text_lower = text.lower()
        found: list[str] = []
        for resource, pattern in _RESOURCE_KEYWORDS.items():
            if re.search(pattern, text_lower):
                found.append(resource)
        return found

    # ------------------------------------------------------------------
    # Dish name extraction
    # 菜名提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dish_name(lines: list[str]) -> str:
        """提取简洁菜名，容忍复制来的网页样板文。"""
        first_candidate: str | None = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip section headers  跳过段标题
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
                first_candidate = stripped
                break

        if first_candidate is None:
            return "Untitled Recipe"

        # A copied article often starts with marketing prose instead of its
        # title. Recover an explicitly named dish from nearby prose before
        # falling back to a bounded first line.
        # 复制来的文章常以营销文案而非标题开头。在回退到有界首行之前，
        # 先从附近文案中恢复显式命名的菜名。
        lower = first_candidate.casefold()
        looks_like_page_prose = (
            len(first_candidate) > 60
            or bool(re.search(r"[.!?…]", first_candidate))
            or lower.startswith(("recipe video", "video above", "recipe notes", "nutrition information"))
        )
        if looks_like_page_prose:
            source = " ".join(line.strip() for line in lines[:12] if line.strip())[:1500]
            english = re.search(
                r"\bThis\s+([A-Z][A-Za-z'’&-]*(?:\s+[A-Z][A-Za-z'’&-]*){0,5})\s+"
                r"(?:is|are|starts?|started|uses|makes|takes|has)\b",
                source,
            )
            if english:
                return english.group(1)[:80]
            chinese = re.search(r"(?:这道|這道|本款|这份|這份)([\u4e00-\u9fff]{2,12})(?:是|的|做法|需|用)", source)
            if chinese:
                return chinese.group(1)[:80]

        return first_candidate[:80]

    @staticmethod
    def _make_recipe_id(dish_name: str) -> str:
        """从菜名生成稳定的 recipe_id。"""
        # Lowercase, replace spaces/special chars with underscores
        # 转小写，把空格 / 特殊字符替换为下划线
        slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", dish_name.lower())
        return f"recipe_{slug[:40]}"

    @staticmethod
    def _detect_language(text: str) -> str:
        """快速检测候选的语言。"""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        if cjk == 0 and latin == 0:
            return "und"
        if cjk > latin:
            return "zho"
        return "eng"

    @staticmethod
    def _extract_servings(text: str) -> Decimal:
        """从菜谱文本提取份数。默认 2。"""
        # "Serves 4", "4 servings", "2人份", "4人"
        match = re.search(r"(?:serves?|servings?|yields?|makes?)\s+(\d+)", text, re.IGNORECASE)
        if match:
            return Decimal(match.group(1))
        match = re.search(r"(\d+)\s*(?:人份|人份量| servings?)", text)
        if match:
            return Decimal(match.group(1))
        # "2-4 servings" — take the middle  "2-4 份" —— 取中间值
        match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*(?:servings?|人份?)", text, re.IGNORECASE)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            return Decimal((lo + hi) // 2)
        return Decimal(2)


# =============================================================================
# Helper: unit normalisation
# 辅助：单位规范化
# =============================================================================


def _normalise_unit(unit: str) -> str:
    """把单位字符串规范化为规范形式。"""
    unit_lower = unit.lower().strip()
    if unit_lower in _CHINESE_UNIT_MAP:
        return _CHINESE_UNIT_MAP[unit_lower]
    # Common English abbreviations  常见英文缩写
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
