# =============================================================================
# 确定性多菜谱解析模块（parsing/recipe_imports）
# -----------------------------------------------------------------------------
# 菜谱导入的“确定性兜底”实现：在不调用 LLM 的情况下，把粘贴的多菜文本
# 切成多个菜谱块，再用既有规则提取器逐个提取。
# 核心函数：
#   - clean_recipe_text          ：切分/提取前统一行尾与空行
#   - split_recipe_blocks        ：按分隔符/标题/空行把多菜文本切块
#   - split_on_markers           ：按 LLM 返回的“菜首行标记”在原文中定位切分
#   - expand_prep_boundaries     ：按 "Ingredients Preparation" 模板标题切块
#   - DeterministicRecipeImportExtractor：走规则提取器逐块提取
# 设计：与 LLM 路径（llm/recipe_importer.py）共享 clean_recipe_text、切分与
#       _candidate_to_draft，保证两条路径产出相同的 RecipeImportDraft 语义。
# =============================================================================

"""Deterministic multi-dish parsing for recipe-import fallback.

菜谱导入兜底的确定性多菜谱解析。
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from cooking_plan_agent.domain.models import ExtractedRecipeCandidate
from cooking_plan_agent.domain.recipe_imports import RecipeImportDraft
from cooking_plan_agent.normalisation.names import clean_dish_name
from cooking_plan_agent.parsing.extractor import RecipeExtractor
from cooking_plan_agent.parsing.preprocess import collapse_blank_lines, normalise_line_endings

_SEPARATOR = re.compile(r"(?m)^\s*(?:-{3,}|={3,})\s*$")
# ↑ 显式分隔符：单独一行的 "---" 或 "==="
# "Recipe: Lemon Pasta", "Recipe 1: Lemon Pasta", "Dish 2 — Stew"…
# ↑ "Recipe: Lemon Pasta"、"Recipe 1: Lemon Pasta"、"Dish 2 — Stew" 等
_RECIPE_HEADING = re.compile(
    r"^\s*(?:recipe|dish|菜谱|食谱|料理|レシピ|요리|receta|receita|recette|rezept|ricetta)"
    r"\s*(?:\d+\s*)?[：:]?\s*(?:[-–—]\s*)?(.*)$",
    re.IGNORECASE,
)
_NON_DISH_RECIPE_SECTION = re.compile(
    r"^(?:video\b|notes?\b|nutrition(?:al)?\b|information\b|tips?\b|cook\s+mode\b)",
    re.IGNORECASE,
)
# ↑ 非菜谱的 "Recipe ..." 段（Recipe Notes / Recipe VIDEO 等）
_MARKDOWN_DISH_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$")
# ↑ Markdown "#" 标题
# "Ingredients Preparation", "I. Ingredients Preparation" — a common pasted
# recipe template heading. Deliberately NOT matching "Ingredients:" sections.
# "Ingredients Preparation"、"I. Ingredients Preparation" —— 常见的粘贴菜谱模板标题。
# 刻意不匹配 "Ingredients:" 段。
_PREP_HEADING = re.compile(r"^\s*(?:[IVX]{1,3}\.\s*)?ingredients?\s+preparation\b", re.IGNORECASE)
# "Main ingredients:" / "- Main ingredients" template section headings (own line).
# "Main ingredients:" / "- Main ingredients" 模板段标题（独占一行）。
_MAIN_INGREDIENTS_HEADING = re.compile(r"^\s*-?\s*(?:main\s+)?ingredients?\s*[：:]*\s*$", re.IGNORECASE)
# "Main ingredients: Fresh shrimp (，…)" — dish name glued after the colon.
# "Main ingredients: Fresh shrimp (，…)" —— 菜名粘连在冒号后。
_MAIN_INGREDIENTS_INLINE = re.compile(r"^\s*-?\s*(?:main\s+)?ingredients?\s*[：:]\s*(.+)$", re.IGNORECASE)
_SERVINGS = re.compile(
    r"\b(?:serves?|servings?|makes?|yield|para|pour|für)\s*(?::|for)?\s*(\d{1,2})\b"
    r"|\b(\d{1,2})\s*(?:servings?|portions?|porciones?|raciones?|porções?|persone?|personen?)\b"
    r"|(?<!\d)(\d{1,2})\s*(?:人份量?|人分|人|份|인분)(?!\d)",
    re.IGNORECASE,
)
# ↑ 份数检测（英文 serves / servings + 中文 人份/人/份 + 韩文 인분）
# Two or more consecutive blank lines = a likely dish boundary when the text
# carries no explicit "---" / "Recipe:" separator (common for copy-paste).
# 两个及以上连续空行 = 当文本没有显式 "---" / "Recipe:" 分隔符时的可能菜品边界（复制粘贴常见）。
_DOUBLE_BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


@runtime_checkable
class RecipeImportExtractor(Protocol):
    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]: ...


def clean_recipe_text(text: str) -> str:
    """在任何切分或提取之前，规范化粘贴的多菜文本。

    Normalise a pasted multi-dish text before any splitting or extraction.

    Reuses the pipeline's deterministic cleaning stages so every recipe-import
    path (LLM, fan-out, rule fallback) sees the same line endings and blank
    lines — otherwise copy-paste noise would break the block heuristics.

    复用流水线的确定性清洗阶段，使每条菜谱导入路径（LLM、fan-out、规则兜底）
    看到相同的行尾与空行 —— 否则复制粘贴噪声会破坏块启发式。
    """

    return collapse_blank_lines(normalise_line_endings(text))


def _is_recipe_heading(line: str) -> bool:
    """判断某行 ``Recipe ...`` 是否真正开启一道菜。

    Return whether a ``Recipe ...`` line starts an actual dish.

    Recipe pages commonly contain headings such as ``Recipe Notes`` and
    boilerplate such as ``Recipe VIDEO above``. Treating those as dish
    boundaries irreversibly splits one recipe before semantic LLM parsing.

    菜谱页常含 ``Recipe Notes`` 这类标题与 ``Recipe VIDEO above`` 这类样板文。
    把它们当作菜品边界会在语义 LLM 解析前不可逆地拆错一道菜。
    """

    match = _RECIPE_HEADING.match(line)
    if match is None:
        return False
    remainder = match.group(1).strip()
    if not remainder or _NON_DISH_RECIPE_SECTION.match(remainder):
        return False
    return not bool(re.search(r"[.!?…]", remainder))


def split_recipe_blocks(text: str, *, split_blank_lines: bool = True) -> tuple[str, ...]:
    """切分常见的多菜谱文本格式，而不猜测菜品内容。

    Split common multi-recipe text formats without guessing dish content.

    Recognised boundaries, in priority order:
      1. Explicit separators ("---" / "===")
      2. "Recipe:" / "Dish:" headings (with or without a number), Markdown
         "#" headings, and "Ingredients Preparation" template headings
      3. Two or more consecutive blank lines (copy-paste style)

    Returns a single block when no reliable boundary is found.

    按优先级识别边界：
      1. 显式分隔符（"---" / "==="）
      2. "Recipe:" / "Dish:" 标题（可带编号）、Markdown "#" 标题、以及
         "Ingredients Preparation" 模板标题
      3. 两个及以上连续空行（复制粘贴风格）

    未找到可靠边界时返回单块。
    """

    separated = tuple(part.strip() for part in _SEPARATOR.split(text) if part.strip())
    if len(separated) > 1:
        return separated

    lines = text.splitlines()
    heading_indexes = [
        index
        for index, line in enumerate(lines)
        if _is_recipe_heading(line) or _MARKDOWN_DISH_HEADING.match(line) or _PREP_HEADING.match(line)
    ]
    if len(heading_indexes) > 1:
        blocks: list[str] = []
        for position, start in enumerate(heading_indexes):
            end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if block:
                blocks.append(block)
        return tuple(blocks)

    # No explicit separator — try the blank-line heuristic as a last resort.
    # 无显式分隔符 —— 最后尝试空行启发式。
    if split_blank_lines:
        blank_split = tuple(part.strip() for part in _DOUBLE_BLANK.split(text) if part.strip())
        if len(blank_split) > 1:
            return blank_split

    return (text.strip(),)


def split_on_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    """按每个 marker 的出现位置（按顺序）切分文本。

    Split text at each marker's occurrence, in order.

    Used by the LLM dish-splitting path: the model returns the first line of
    every dish and this locates those substrings in the original text, so no
    text is ever rewritten or lost. Substring (not line) matching also splits
    pasted text where the next dish's heading is glued onto the previous
    dish's last line (e.g. "...serve. Ingredients Preparation").

    用于 LLM 切菜路径：模型返回每道菜的首行，本函数在原文中定位这些子串，
    因此文本绝不重写或丢失。子串（非行）匹配也能切分“下一道菜标题粘连在
    上一道菜末行”的粘贴文本（如 "...serve. Ingredients Preparation"）。
    """

    lower = text.lower()
    starts: list[int] = []
    cursor = 0
    for marker in markers:
        needle = marker.strip().lower()
        if not needle:
            continue
        position = lower.find(needle, cursor)
        if position < 0:
            continue
        starts.append(position)
        cursor = position + len(needle)
    if len(starts) < 2:
        return (text,)

    blocks: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return tuple(blocks)


# "ingredients preparation" (optionally "I. Ingredients Preparation") is the
# pasted-template dish heading itself. Splitting on it as a SUBSTRING (not a
# line start) also catches headings glued to the previous dish's last line.
# "ingredients preparation"（可选带 "I. Ingredients Preparation"）本身就是粘贴模板的
# 菜品标题。把它作为子串（而非行首）切分，也能捕获粘连在上一道菜末行的标题。
_PREP_BOUNDARY = re.compile(r"(?:[IVX]{1,3}\.\s*)?ingredients?\s+preparation\b", re.IGNORECASE)


def expand_prep_boundaries(block: str) -> tuple[str, ...]:
    """按 "Ingredients Preparation" 标题出现的每一处切分一个块。

    Split a block on "Ingredients Preparation" headings wherever they occur.

    Deterministic coarse cut for template pastes: each occurrence of the
    heading starts a new dish. Glued headings ("...serve. Ingredients
    Preparation") are still caught because matching is substring-based.
    Returns the input unchanged when there is at most one occurrence.

    模板粘贴的确定性粗切：标题每出现一次就开启一道新菜。粘连标题
    （"...serve. Ingredients Preparation"）仍能被捕获，因为匹配基于子串。
    出现次数不超过一次时原样返回输入。
    """

    matches = list(_PREP_BOUNDARY.finditer(block))
    if len(matches) <= 1:
        return (block,)
    starts = [match.start() for match in matches]
    # Segment k runs [0|starts[k], starts[k+1]|len): the first segment keeps
    # any content before the first heading, later segments start at their own
    # heading so "…serve. Ingredients Preparation" never leaks into a dish.
    # 第 k 段为 [0|starts[k], starts[k+1]|len)：第一段保留第一个标题之前的内容，
    # 后续段从各自标题开始，使 "…serve. Ingredients Preparation" 绝不泄漏进某道菜。
    seg_starts = [0, *starts[1:]]
    seg_ends = [*starts[1:], len(block)]
    parts: list[str] = []
    for start, end in zip(seg_starts, seg_ends, strict=True):
        part = block[start:end].strip()
        if part:
            parts.append(part)
    return tuple(parts)


def _normalise_heading(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    # "Recipe: X" / "# X" → keep X as the dish name.
    # "Recipe: X" / "# X" → 保留 X 作为菜名。
    match = (_RECIPE_HEADING.match(lines[0]) if _is_recipe_heading(lines[0]) else None) or _MARKDOWN_DISH_HEADING.match(
        lines[0]
    )
    if match:
        lines[0] = match.group(1).strip()
    # Drop pasted template headings ("Ingredients Preparation",
    # "Main ingredients:") so the extractor sees the actual dish name; when
    # the name is glued to the heading line ("Main ingredients: Crab legs")
    # keep only the name.
    # 丢弃粘贴模板标题（"Ingredients Preparation"、"Main ingredients:"），使提取器
    # 看到真正的菜名；当菜名粘连在标题行（"Main ingredients: Crab legs"）时只保留菜名。
    while lines:
        stripped = lines[0].strip()
        if _PREP_HEADING.match(stripped) or _MAIN_INGREDIENTS_HEADING.match(stripped):
            lines.pop(0)
            continue
        inline = _MAIN_INGREDIENTS_INLINE.match(stripped)
        if inline:
            name = inline.group(1).strip()
            if name:
                lines[0] = name
            else:
                lines.pop(0)
            continue
        break
    return "\n".join(lines)


def _explicit_servings(block: str) -> int | None:
    match = _SERVINGS.search(block)
    if not match:
        return None
    value = int(next(group for group in match.groups() if group is not None))
    return value if 1 <= value <= 50 else None


def _candidate_to_draft(
    index: int,
    raw_block: str,
    candidate: ExtractedRecipeCandidate,
) -> RecipeImportDraft:
    """把完整提取候选映射为部分导入草稿。

    Map a full extracted candidate into a partial import draft.

    Shared by the deterministic extractor and the LLM fan-out path so both
    produce identical ``RecipeImportDraft`` semantics: servings are only
    recorded when the raw text explicitly states them (never a rule default),
    and free-text names are surfaced verbatim.

    由确定性提取器与 LLM fan-out 路径共享，使两者产出相同的 RecipeImportDraft 语义：
    份数仅在原文显式声明时才记录（绝不用规则默认值），自由文本名称原样呈现。
    """

    name = clean_dish_name(candidate.dish_name) if candidate.dish_name else None
    if name in {"Untitled Recipe", "Untitled"}:
        name = None
    return RecipeImportDraft(
        draft_id=f"dish-{index}",
        name=name,
        servings=_explicit_servings(raw_block),
        ingredients=tuple(
            ingredient.raw_text.strip() for ingredient in candidate.ingredients if ingredient.raw_text.strip()
        )[:100],
        steps=tuple(step.instruction.strip() for step in candidate.steps if step.instruction.strip())[:100],
    )


class DeterministicRecipeImportExtractor:
    """通过既有规则提取器解析已识别的菜谱段。"""

    def __init__(self, recipe_extractor: RecipeExtractor | None = None) -> None:
        self._recipe_extractor = recipe_extractor or RecipeExtractor()

    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]:
        cleaned = clean_recipe_text(text)
        drafts: list[RecipeImportDraft] = []
        for index, raw_block in enumerate(split_recipe_blocks(cleaned), start=1):
            block = _normalise_heading(raw_block)
            candidate = await self._recipe_extractor.extract(block)
            drafts.append(_candidate_to_draft(index, raw_block, candidate))
        return tuple(drafts)
