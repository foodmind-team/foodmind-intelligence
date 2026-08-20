# =============================================================================
# LLM 多菜谱导入提取器模块（llm/recipe_importer）
# -----------------------------------------------------------------------------
# 实现“多菜谱导入”的 LLM 提取。导入边界复用了 Agent 的完整自然语言流水线：
#   1. clean_recipe_text    —— 确定性清洗（行尾、空行）
#   2. split_recipe_blocks  —— 确定性多菜切分
#   3. 多块输入 → 按菜并发 fan-out 到 LLMRecipeExtractor（每块是短小、全结构化的
#      提取；失败仅对该块降级到规则提取器）
#   4. 单块输入 → 对整个 recipes 数组做一次 LLM 调用，并抬高输出预算，
#      避免多菜 JSON 在数组中间被截断
# 所有路径都回退到 DeterministicRecipeImportExtractor 而非报错，
# 因此 provider 故障绝不阻塞交互式导入流程。
# 关键点：多菜识别先做“语义切分”（LLM 识别每道菜的首行标记），再并发提取；
#        英文规范化（_ensure_english）是“展示级”的最终闸门，失败不回滚结构数据。
# =============================================================================

"""LLM-backed multi-dish recipe import extraction.

基于 LLM 的多菜谱导入提取。

The import boundary reuses the agent's full natural-language pipeline:

  1. ``clean_recipe_text``  — deterministic cleaning (line endings, blanks)
  2. ``split_recipe_blocks`` — deterministic multi-dish splitting
  3. multi-block input → per-dish ``LLMRecipeExtractor`` fan-out (each block is
     a short, fully-structured extraction; failures degrade to the rule
     extractor for that block only)
  4. single-block input → one LLM call for the whole ``recipes`` array with a
     raised output budget, so multi-dish JSON is never truncated mid-array

Every path falls back to ``DeterministicRecipeImportExtractor`` instead of
failing, so a provider outage never blocks the interactive import flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from typing import Any

from cooking_plan_agent.application.recipe_import_service import InvalidRecipeImportAnswers
from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.recipe_imports import RecipeImportAnswer, RecipeImportDraft, RecipeImportQuestion
from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.llm.extractor import LLMRecipeExtractor
from cooking_plan_agent.normalisation.names import clean_dish_name
from cooking_plan_agent.parsing.extractor import RecipeExtractor
from cooking_plan_agent.parsing.recipe_imports import (
    DeterministicRecipeImportExtractor,
    _candidate_to_draft,
    _normalise_heading,
    clean_recipe_text,
    expand_prep_boundaries,
    split_on_markers,
    split_recipe_blocks,
)

_SYSTEM_PROMPT = (
    "You extract one or more recipes from user text written in any language. Return one JSON object only with "
    "a recipes array. Each recipe must contain exactly: name (string or null), servings "
    "(whole number or null), ingredients (array of strings), and steps (array of strings). "
    "Translate name, ingredients, and steps into clear English before returning them. All recipe strings in "
    "the JSON response must be English even when the source is multilingual. Preserve quantities, units, "
    "temperatures, cooking times, and proper nouns accurately. "
    "Count distinct finished dishes, not page sections or component mixtures. Ingredient groups such as a seasoning "
    "mix, sauce, marinade, or topping belong to their parent dish. Recipe notes, substitutions, nutrition, cook-mode "
    "text, video references, and serving tips are metadata and must NEVER become separate recipes. "
    "Preserve input order. name must be a SHORT conventional dish title only — ignore page introductions and strip "
    "site boilerplate, quantities, units, "
    "parenthetical notes, and preparation instructions (e.g. 'Fresh Shrimp', not 'Fresh shrimp "
    "(remove head, tail, and thread)'; 'chicken wings', not '15 chicken wings'). "
    "CRITICAL: when the text describes MULTIPLE distinct dishes — "
    "separated by '---', 'Recipe:' headings, blank lines, or simply listed one after another — "
    "you MUST return one recipe object per dish. Never merge two dishes into a single recipe "
    "object. A dish name may be inferred only from an explicit reference in the text (for example, 'This Jambalaya' "
    "means the title is 'Jambalaya'); never copy an introductory sentence as the name. Never invent a serving count, "
    "ingredient, or step that the user did not provide; use null or an empty array when required information is missing."
)
# ↑ 主提取提示词：从任意语言文本提取一道或多道菜，强制英文输出、一菜一对象、禁止臆造

_ANSWER_SYSTEM_PROMPT = (
    "Translate recipe clarification answer values into clear English. Return one JSON object only with an "
    "answers array containing exactly question_id and value for every supplied answer. Preserve question_id, "
    "numbers, quantities, units, temperatures, cooking times, line breaks, and list item boundaries. Do not "
    "add or remove recipe facts. Every textual value must be English. For a servings answer, convert a number "
    "written in words or another numeral system to ASCII digits only."
)
# ↑ 答案翻译提示词：把澄清答案翻译成英文，保持 question_id 与事实不变

_SPLIT_SYSTEM_PROMPT = (
    "You split a pasted cooking text into its separate dishes. Return one JSON object only with "
    "a dishes array. Each element is the exact first line of one dish, copied verbatim from the "
    "input — for example 'Recipe: Lemon Pasta', 'Ingredients Preparation', or a dish name line "
    "such as 'Fried Spare Ribs'. A marker must begin a distinct finished dish. Never use section headings such as "
    "'Creole Seasoning Mix', 'Sauce', 'Marinade', 'Recipe Notes', 'Nutrition Information', 'Cook Mode', or a video/site "
    "introduction as dish markers; those belong to the surrounding recipe. Cover every dish in the text in order. If there is only one "
    "dish, return one element. Never rewrite, translate, or invent lines that do not appear "
    "in the input. IMPORTANT: a dish's name may be glued onto the previous dish's last line "
    "after an ingredient, e.g. '...Cooking oil Fried Spare Ribs' — when you see such a new "
    "dish name, still report it; a marker does not have to start at a line boundary."
)
# ↑ 语义切分提示词：让 LLM 识别每道菜的“首行标记”，用于按菜切分文本

_TRANSLATION_SYSTEM_PROMPT = (
    "You normalize already-extracted recipe drafts into English. Return one JSON object only with a recipes "
    "array. Preserve each draft_id, servings value, list order, quantities, units, temperatures, times, and "
    "proper nouns. Translate name, every ingredient, and every step into clear English. Do not add, remove, "
    "merge, or reinterpret recipe facts. Every letter in user-visible fields must use Latin script."
)
# ↑ 英文规范化提示词：把已提取的草稿翻译成英文，保留 draft_id / 份数 / 顺序 / 数量

logger = logging.getLogger(__name__)


def _contains_non_latin_letters(value: str) -> bool:
    """判断字符串是否含非拉丁字母（用于检测未翻译的多语言文本）。"""
    return any(character.isalpha() and "LATIN" not in unicodedata.name(character, "") for character in value)


def _needs_english_normalisation(drafts: tuple[RecipeImportDraft, ...]) -> bool:
    """判断草稿集是否还含有非拉丁文本（需要英文规范化）。"""
    return any(
        _contains_non_latin_letters(value)
        for draft in drafts
        for value in (draft.name or "", *draft.ingredients, *draft.steps)
    )


class LLMRecipeImportExtractor:
    """提取有界数量的部分导入草稿，并带安全兜底。"""

    def __init__(
        self,
        client: LLMClient,
        fallback: DeterministicRecipeImportExtractor | None = None,
        *,
        timeout_seconds: float = 20.0,
        max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or DeterministicRecipeImportExtractor()
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        # Per-dish fan-out reuses the workflow's full-field extractor, so every
        # dish benefits from the same rich prompt and rule degradation as the
        # main cooking-plan pipeline.
        # 按菜 fan-out 复用工作流的全字段提取器，使每道菜享受与主烹饪计划流水线
        # 相同的丰富提示词与规则降级。
        self._dish_extractor = LLMRecipeExtractor(client, translate_to_english=True)

    async def normalise_answers(
        self,
        questions: tuple[RecipeImportQuestion, ...],
        answers: tuple[RecipeImportAnswer, ...],
    ) -> tuple[RecipeImportAnswer, ...]:
        """翻译澄清答案的自由文本，同时保留其 ID。"""

        field_by_id = {question.question_id: question.field_path for question in questions}
        if not answers:
            return answers
        request_payload = {
            "answers": [
                {
                    "question_id": answer.question_id,
                    "field_path": field_by_id.get(answer.question_id, "text"),
                    "value": answer.value,
                }
                for answer in answers
            ]
        }
        try:
            payload = await asyncio.wait_for(
                self._client.chat_json(
                    [
                        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
                    ],
                    max_tokens=self._max_output_tokens,
                ),
                timeout=self._timeout_seconds,
            )
            translated_items = payload.get("answers")
            if not isinstance(translated_items, list):
                raise LLMError("Recipe answer translation did not contain an answers list")
            translated = {
                str(item.get("question_id")): str(item.get("value", "")).strip()
                for item in translated_items
                if isinstance(item, dict) and item.get("question_id") and str(item.get("value", "")).strip()
            }
            expected_ids = {answer.question_id for answer in answers}
            if len(translated_items) != len(expected_ids) or set(translated) != expected_ids:
                raise LLMError("Recipe answer translation changed the answer identifiers")
        except (TimeoutError, LLMError, TypeError, ValueError) as exc:
            raise InvalidRecipeImportAnswers(
                "Recipe answer translation is temporarily unavailable. Please try again."
            ) from exc

        return tuple(
            RecipeImportAnswer(question_id=answer.question_id, value=translated.get(answer.question_id, answer.value))
            for answer in answers
        )

    async def extract(self, text: str) -> tuple[RecipeImportDraft, ...]:
        """把粘贴文本提取为若干菜谱草稿。"""
        cleaned = clean_recipe_text(text)
        # Deterministic coarse cut: separators, headings, blank lines, then
        # "Ingredients Preparation" template boundaries (substring-based, so
        # glued headings like "...serve. Ingredients Preparation" still cut).
        # 确定性粗切：分隔符、标题、空行，然后是 "Ingredients Preparation" 模板边界
        # （基于子串，因此像 "...serve. Ingredients Preparation" 这种粘连标题仍能切开）。
        coarse: list[str] = []
        # Blank lines are presentation, not a reliable dish boundary. Leave
        # them to the semantic splitter; deterministic fallback can still use
        # the legacy blank-line heuristic when the provider is unavailable.
        # 空行是排版，不是可靠的菜品边界。交给语义切分器处理；
        # 确定性兜底在 provider 不可用时仍可用旧版空行启发式。
        for block in split_recipe_blocks(cleaned, split_blank_lines=False):
            coarse.extend(expand_prep_boundaries(block))
        blocks = tuple(coarse)

        # LLM semantic splitting is the primary boundary detector: it handles
        # heading-less dishes and other layouts the deterministic rules cannot
        # see. Running it per block keeps every call short (a full 6-dish paste
        # times out in one shot) and lets still-merged blocks expand. Blocks
        # are independent, so they split concurrently under the LLM semaphore.
        # LLM 语义切分是主要的边界检测器：它能处理无标题的菜以及其他确定性规则
        # 看不到的布局。按块运行使每次调用保持短小（一整段 6 道菜的粘贴会一次超时），
        # 并让仍被合并的块得以展开。块彼此独立，因此在 LLM 信号量下并发切分。
        settings = get_settings()
        semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

        async def _split_one(block: str) -> tuple[str, ...]:
            async with semaphore:
                try:
                    sub_blocks = await self._split_with_llm(block)
                    return sub_blocks if len(sub_blocks) > 1 else (block,)
                except (TimeoutError, LLMError, TypeError, ValueError):
                    return (block,)

        nested = await asyncio.gather(*(_split_one(block) for block in blocks))
        blocks = tuple(part for group in nested for part in group)

        if len(blocks) > 1:
            # Recognisable multi-dish text → extract each block independently.
            # 可识别的多菜文本 → 独立提取每个块。
            try:
                drafts = await self._extract_multi(blocks)
            except (TimeoutError, LLMError, TypeError, ValueError):
                drafts = await self._fallback.extract(cleaned)
            return await self._ensure_english(drafts)
        # Single block → whole-text recipes-array extraction.
        # 单块 → 对整个文本做 recipes 数组提取。
        try:
            drafts = await self._extract_single(cleaned)
        except (TimeoutError, LLMError, TypeError, ValueError):
            drafts = await self._fallback.extract(cleaned)
        return await self._ensure_english(drafts)

    async def _ensure_english(self, drafts: tuple[RecipeImportDraft, ...]) -> tuple[RecipeImportDraft, ...]:
        """把英文规范化作为“有界最终闸门”重试，而不是持久化混合文字草稿。"""
        if not _needs_english_normalisation(drafts):
            return drafts
        request_payload = {"recipes": [draft.model_dump() for draft in drafts]}
        try:
            payload = await asyncio.wait_for(
                self._client.chat_json(
                    [
                        {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
                    ],
                    max_tokens=self._max_output_tokens,
                ),
                timeout=self._timeout_seconds,
            )
            values = payload.get("recipes")
            if not isinstance(values, list) or len(values) != len(drafts):
                raise LLMError("Recipe translation changed the draft count")
            translated: list[RecipeImportDraft] = []
            for index, (previous, value) in enumerate(zip(drafts, values, strict=True), start=1):
                if not isinstance(value, dict) or value.get("draft_id") != previous.draft_id:
                    raise LLMError("Recipe translation changed a draft identifier")
                candidate = self._draft(index, value).model_copy(update={"draft_id": previous.draft_id})
                if candidate.servings != previous.servings:
                    raise LLMError("Recipe translation changed a serving count")
                translated.append(candidate)
            result = tuple(translated)
            if _needs_english_normalisation(result):
                raise LLMError("Recipe translation still contains non-Latin text")
            return result
        except (TimeoutError, LLMError, TypeError, ValueError) as exc:
            # Import availability takes precedence over a presentation-only
            # translation pass. The drafts have already been structurally
            # extracted, and the review screen lets the user inspect them
            # before anything is persisted. Returning them is safer than
            # discarding a valid multilingual recipe because the optional
            # English normalisation provider had a transient failure.
            # 导入可用性优先于“仅展示”的翻译步骤。草稿已完成结构化提取，
            # 审阅界面允许用户在持久化前检查。返回它们比因可选的英文规范化
            # provider 瞬时故障而丢弃一份有效的多语言菜谱更安全。
            logger.warning(
                "Recipe-import English normalisation failed; returning reviewable source-language drafts",
                exc_info=exc,
            )
            return drafts

    async def _split_with_llm(self, text: str) -> tuple[str, ...]:
        """让 LLM 给出每道菜的首行，再按这些行切分。

        Returns a single block when the model reports one dish or its markers
        cannot be located, so the caller falls through to whole-text parsing.

        当模型只报告一道菜或标记无法定位时返回单块，调用方将落到整文本解析。
        """
        payload = await asyncio.wait_for(
            self._client.chat_json(
                [
                    {"role": "system", "content": _SPLIT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=self._max_output_tokens,
            ),
            # The split reply is tiny — never let it consume the whole budget.
            # Per-block calls stay well under this; whole-text pastes may not.
            # 切分回复很小 —— 绝不让它耗尽整个预算。按块调用远低于此；整文本粘贴可能不会。
            timeout=min(self._timeout_seconds, 6.0),
        )
        markers = payload.get("dishes")
        if not isinstance(markers, list):
            raise LLMError("Dish split response did not contain a dishes list")
        clean_markers = tuple(str(marker).strip() for marker in markers if isinstance(marker, str) and marker.strip())
        if len(clean_markers) < 2:
            return (text,)
        blocks = split_on_markers(text, clean_markers)
        return blocks if len(blocks) > 1 else (text,)

    async def _extract_multi(self, blocks: tuple[str, ...]) -> tuple[RecipeImportDraft, ...]:
        settings = get_settings()
        semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

        async def _one(index: int, raw_block: str) -> RecipeImportDraft:
            block = _normalise_heading(raw_block)
            async with semaphore:
                try:
                    candidate = await asyncio.wait_for(
                        self._dish_extractor.extract(block),
                        # DeepSeek completes a single dish well under this;
                        # a hung provider must not stall the whole import.
                        # DeepSeek 单道菜的完成时间远低于此；卡死的 provider 绝不能拖垮整个导入。
                        timeout=min(self._timeout_seconds, 10.0),
                    )
                except (TimeoutError, LLMError, TypeError, ValueError):
                    # Progressive degradation: a single bad block must not
                    # fail the whole import — use the rule extractor for it.
                    # 渐进降级：单个坏块绝不能使整个导入失败 —— 对它用规则提取器。
                    candidate = await RecipeExtractor().extract(block)
            return _candidate_to_draft(index, raw_block, candidate)

        drafts = await asyncio.wait_for(
            asyncio.gather(*(_one(index, block) for index, block in enumerate(blocks, start=1))),
            timeout=settings.llm_overall_timeout_seconds,
        )
        return tuple(drafts)

    async def _extract_single(self, text: str) -> tuple[RecipeImportDraft, ...]:
        payload = await asyncio.wait_for(
            self._client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=self._max_output_tokens,
            ),
            timeout=self._timeout_seconds,
        )
        recipes = payload.get("recipes")
        if not isinstance(recipes, list) or not recipes:
            raise LLMError("Recipe import response did not contain recipes")
        return tuple(
            self._draft(index, item, source_text=text)
            for index, item in enumerate(recipes, start=1)
            if isinstance(item, dict)
        )

    @staticmethod
    def _draft(index: int, value: dict[str, Any], *, source_text: str = "") -> RecipeImportDraft:
        """把 LLM 输出的单个菜谱对象转换为 RecipeImportDraft（防御式）。"""
        raw_servings = value.get("servings")
        servings: int | None = None
        if isinstance(raw_servings, int) and not isinstance(raw_servings, bool) and 1 <= raw_servings <= 50:
            servings = raw_servings
        name_value = value.get("name")
        name = clean_dish_name(str(name_value)) if isinstance(name_value, str) and name_value.strip() else None
        if name and (len(name) > 80 or re.search(r"[.!?…]", name)) and source_text:
            # Enforce the public short-title contract even if the provider
            # echoes webpage prose. The deterministic extractor recognises
            # explicit references such as "This Jambalaya is ...".
            # 即使 provider 回显了网页散文，也要强制执行“简短标题”契约。
            # 确定性提取器能识别 "This Jambalaya is ..." 这类显式引用。
            name = RecipeExtractor._extract_dish_name(source_text.splitlines())
        name = name[:80] if name else None
        ingredients = tuple(
            str(item).strip()[:500] for item in value.get("ingredients", []) if isinstance(item, str) and item.strip()
        )[:100]
        steps = tuple(
            str(item).strip()[:1000] for item in value.get("steps", []) if isinstance(item, str) and item.strip()
        )[:100]
        return RecipeImportDraft(
            draft_id=f"dish-{index}",
            name=name,
            servings=servings,
            ingredients=ingredients,
            steps=steps,
        )
