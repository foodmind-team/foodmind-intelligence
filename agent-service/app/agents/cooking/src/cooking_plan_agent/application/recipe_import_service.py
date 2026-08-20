"""Application service for multi-dish recipe import clarification."""

from __future__ import annotations

import re
from typing import Protocol

from cooking_plan_agent.config.settings import get_settings
from cooking_plan_agent.domain.recipe_imports import (
    ParseRecipeImportRequest,
    ParseRecipeImportResponse,
    RecipeImportAnswer,
    RecipeImportDraft,
    RecipeImportQuestion,
    RecipeImportStatus,
)
from cooking_plan_agent.parsing.recipe_imports import RecipeImportExtractor

# 列表前缀（如 "-"、"*"、"•"、"1."、"2)"），用于清洗用户逐行粘贴的食材/步骤
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
# 草稿未填份数时的默认份数
_DEFAULT_SERVINGS = 2


class InvalidRecipeImportAnswers(ValueError):
    """Raised when answers do not match the current structured questions."""

    # 答案与当前结构化问题不匹配时抛出的业务异常（信号类）


class RecipeImportAnswerNormaliser(Protocol):
    """Converts free-text clarification answers into English."""

    # 结构化接口：把自由文本答案规范化（如翻译成英文），依赖倒置边界

    async def normalise_answers(
        self,
        questions: tuple[RecipeImportQuestion, ...],
        answers: tuple[RecipeImportAnswer, ...],
    ) -> tuple[RecipeImportAnswer, ...]: ...


class ParseRecipeImportService:
    """Extract drafts, apply field-scoped answers, and produce questions."""

    def __init__(
        self,
        extractor: RecipeImportExtractor,
        answer_normaliser: RecipeImportAnswerNormaliser | None = None,
    ) -> None:
        self._extractor = extractor
        self._answer_normaliser = answer_normaliser

    async def execute(self, request: ParseRecipeImportRequest) -> ParseRecipeImportResponse:
        # 阶段一：取草稿——优先用续聊快照，否则从原始文本抽取
        drafts = request.drafts or await self._extractor.extract(request.text)
        settings = get_settings()
        if not drafts:
            # 抽取为空时兜底生成一个空草稿，避免流程中断
            drafts = (RecipeImportDraft(draft_id="dish-1"),)
        if len(drafts) > settings.max_recipe_count:
            # 超出单次导入上限，直接拒绝
            raise InvalidRecipeImportAnswers(
                f"A maximum of {settings.max_recipe_count} recipes can be imported at once."
            )
        # 份数为空则补默认值 2
        drafts = tuple(
            draft.model_copy(update={"servings": _DEFAULT_SERVINGS}) if draft.servings is None else draft
            for draft in drafts
        )

        # 阶段二：派生问题并校验续聊快照一致性
        derived_questions = self._questions(drafts)
        current_questions = request.questions or derived_questions
        if request.questions and not self._same_questions(request.questions, derived_questions):
            # 快照过期/不一致，防止答案张冠李戴
            raise InvalidRecipeImportAnswers("The recipe-import resume snapshot is stale or inconsistent.")
        # 合法问题表 + 用户答案表，均以 question_id 为键
        allowed = {question.question_id: question for question in current_questions}
        answers = {answer.question_id: answer.value for answer in request.answers}
        unknown = sorted(set(answers) - set(allowed))
        if unknown:
            # 存在无法匹配到任何问题的答案
            raise InvalidRecipeImportAnswers("One or more answers do not match the current questions.")
        # 阶段三：挑出需要规范化的答案（排除已是合法数字份数的 servings）
        answers_requiring_normalisation = tuple(
            answer
            for answer in request.answers
            if not self._is_valid_numeric_servings_answer(allowed[answer.question_id], answer.value)
        )
        if answers_requiring_normalisation and self._answer_normaliser is not None:
            normalised_answers = await self._answer_normaliser.normalise_answers(
                current_questions,
                answers_requiring_normalisation,
            )
            # 用规范化结果覆盖，再合并回无需规范化的答案（如 servings）
            answers = {answer.question_id: answer.value for answer in normalised_answers}
            answers.update(
                (answer.question_id, answer.value)
                for answer in request.answers
                if answer not in answers_requiring_normalisation
            )

        # 阶段四：把答案写回草稿，重新派生问题以判定最终状态
        updated = tuple(self._apply_answers(draft, allowed, answers) for draft in drafts)
        questions = self._questions(updated)
        # 仍有缺失字段 → 继续澄清；否则就绪
        status = RecipeImportStatus.NEEDS_CLARIFICATION if questions else RecipeImportStatus.READY
        return ParseRecipeImportResponse(status=status, drafts=updated, questions=questions)

    @staticmethod
    def _is_valid_numeric_servings_answer(question: RecipeImportQuestion, value: str) -> bool:
        # 仅当字段是 servings、值为纯 ASCII 十进制且在 1~50 之间，才算合法数字份数
        if question.field_path != "servings":
            return False
        stripped = value.strip()
        return stripped.isascii() and stripped.isdecimal() and 1 <= int(stripped) <= 50

    @staticmethod
    def _same_questions(
        supplied: tuple[RecipeImportQuestion, ...],
        derived: tuple[RecipeImportQuestion, ...],
    ) -> bool:
        # Accept pre-deployment snapshots that still contain a servings
        # question. Missing servings are now a deterministic operational
        # default, but an answer already entered by the user should still win.
        # 比较时过滤掉 servings：servings 现已为确定性默认值，不再生成问题；
        # 但旧部署快照可能残留 servings 问题，用户已输入的答案仍应优先保留。
        return {
            (question.question_id, question.draft_id, question.field_path)
            for question in supplied
            if question.field_path != "servings"
        } == {(question.question_id, question.draft_id, question.field_path) for question in derived}

    @staticmethod
    def _apply_answers(
        draft: RecipeImportDraft,
        allowed: dict[str, RecipeImportQuestion],
        answers: dict[str, str],
    ) -> RecipeImportDraft:
        updates: dict[str, object] = {}
        for question_id, value in answers.items():
            question = allowed[question_id]
            # 答案只作用于其所属草稿，避免多道菜答案互相串
            if question.draft_id != draft.draft_id:
                continue
            if question.field_path == "name":
                # 菜名：去空格、截断 160 字符，空则置 None
                updates["name"] = value.strip()[:160] or None
            elif question.field_path == "servings":
                try:
                    servings = int(value)
                except ValueError:
                    continue
                # 仅 1~50 的份数才写入
                if 1 <= servings <= 50:
                    updates["servings"] = servings
            elif question.field_path in {"ingredients", "steps"}:
                # 食材单项限 500 字符、步骤单项限 1000 字符；去掉列表符号后截前 100 项
                limit = 500 if question.field_path == "ingredients" else 1000
                items = tuple(
                    cleaned[:limit]
                    for line in re.split(r"[\n;]+", value)
                    if (cleaned := _LIST_PREFIX.sub("", line).strip())
                )[:100]
                if items:
                    updates[question.field_path] = items
        return draft.model_copy(update=updates)

    @staticmethod
    def _questions(drafts: tuple[RecipeImportDraft, ...]) -> tuple[RecipeImportQuestion, ...]:
        questions: list[RecipeImportQuestion] = []
        for draft in drafts:
            label = draft.name or "this dish"
            # 只对缺失字段提问：缺菜名问菜名、缺食材问食材、缺步骤问步骤（servings 有默认值故不问）
            if not draft.name:
                questions.append(ParseRecipeImportService._question(draft, "name", "What is the dish name?"))
            if not draft.ingredients:
                questions.append(
                    ParseRecipeImportService._question(
                        draft,
                        "ingredients",
                        f"List the ingredients for {label}, with one ingredient per line.",
                    )
                )
            if not draft.steps:
                questions.append(
                    ParseRecipeImportService._question(
                        draft,
                        "steps",
                        f"List the cooking steps for {label}, with one step per line.",
                    )
                )
        return tuple(questions)

    @staticmethod
    def _question(draft: RecipeImportDraft, field_path: str, prompt: str) -> RecipeImportQuestion:
        # question_id 由 "{draft_id}:{field_path}" 构成，唯一且可回溯到具体草稿字段
        return RecipeImportQuestion(
            question_id=f"{draft.draft_id}:{field_path}",
            draft_id=draft.draft_id,
            field_path=field_path,
            prompt=prompt,
        )
