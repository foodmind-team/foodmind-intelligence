# =============================================================================
# 菜谱导入契约模块（domain/recipe_imports）
# -----------------------------------------------------------------------------
# 本文件定义“自然语言菜谱导入”功能的类型化契约，包括：
#   - RecipeImportStatus        ：导入流程的状态（需要澄清 / 就绪）
#   - RecipeImportDraft         ：解析过程中的草稿快照（可多菜拆分）
#   - RecipeImportQuestion      ：需要用户澄清的字段级问题
#   - ParseRecipeImportRequest  ：解析请求（支持“续答”式多轮交互）
#   - ParseRecipeImportResponse ：解析响应
# 设计要点：多轮澄清采用“草稿 + 问题”快照回传机制，保证无状态服务的幂等续答。
# =============================================================================

"""Typed contract for natural-language recipe imports.

自然语言菜谱导入的类型化契约。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from cooking_plan_agent.domain.models import StrictModel


class RecipeImportStatus(StrEnum):
    """菜谱导入状态：导入流程当前所处的阶段。"""

    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    # ↑ 需要澄清：存在缺失 / 歧义字段，需用户补充
    READY = "READY"
    # ↑ 就绪：草稿已完整，可直接进入后续处理


class RecipeImportAnswer(StrictModel):
    """菜谱导入答案：用户对某个澄清问题的作答。"""

    question_id: str = Field(min_length=1, max_length=160)
    # ↑ 所回答的问题标识（对应 RecipeImportQuestion.question_id）
    value: str = Field(min_length=1, max_length=20_000)
    # ↑ 答案内容（允许较长，因为可能是大段补充说明）


class RecipeImportDraft(StrictModel):
    """菜谱导入草稿：解析过程中积累的中间快照。

    一个请求可能拆分出多个草稿（多菜合并场景），每个草稿对应一道菜。
    """

    draft_id: str = Field(min_length=1, max_length=80)
    # ↑ 草稿唯一标识
    name: str | None = Field(default=None, max_length=160)
    # ↑ 菜名（可能尚未确定，故可为 None）
    servings: int | None = Field(default=None, ge=1, le=50)
    # ↑ 份数（1~50，可为 None 表示待澄清）
    ingredients: tuple[str, ...] = Field(default=(), max_length=100)
    # ↑ 食材列表（最多 100 条）
    steps: tuple[str, ...] = Field(default=(), max_length=100)
    # ↑ 步骤列表（最多 100 步，防止多菜合并导致超长）


class RecipeImportQuestion(StrictModel):
    """菜谱导入问题：需要用户澄清的字段级问题。"""

    question_id: str = Field(min_length=1, max_length=160)
    # ↑ 问题唯一标识
    draft_id: str = Field(min_length=1, max_length=80)
    # ↑ 该问题针对的草稿
    field_path: str = Field(min_length=1, max_length=80)
    # ↑ 问题指向的字段路径（如 "name" / "servings"）
    prompt: str = Field(min_length=1, max_length=500)
    # ↑ 提示语（呈现给用户）
    response_type: str = "TEXT"
    # ↑ 响应类型（默认 TEXT 自由文本）
    required: bool = True
    # ↑ 是否必答
    suggested_value: str | None = Field(default=None, max_length=500)
    # ↑ 建议值（供用户一键采纳）


class ParseRecipeImportRequest(StrictModel):
    """菜谱导入解析请求。

    支持“续答”式多轮交互：后续请求会带上之前的 answers / drafts / questions
    快照，使无状态服务能基于历史上下文继续解析，而不是从头开始。
    """

    request_id: str = Field(min_length=1, max_length=128)
    # ↑ 请求唯一标识
    text: str = Field(min_length=1, max_length=100_000)
    # ↑ 原始菜谱文本
    answers: tuple[RecipeImportAnswer, ...] = Field(default=(), max_length=24)
    # ↑ 用户已作答的答案集（最多 24 条）
    drafts: tuple[RecipeImportDraft, ...] = Field(default=(), max_length=6)
    # ↑ 既有草稿快照（最多 6 个，对应多菜拆分）
    questions: tuple[RecipeImportQuestion, ...] = Field(default=(), max_length=24)
    # ↑ 既有问题快照（最多 24 条）

    @model_validator(mode="after")
    def valid_resume_snapshot(self) -> ParseRecipeImportRequest:
        """校验续答快照的完整性：ID 不可重复，且带问题必带草稿。"""
        answer_identifiers = [answer.question_id for answer in self.answers]
        if len(answer_identifiers) != len(set(answer_identifiers)):
            raise ValueError("Duplicate recipe-import question IDs are not allowed.")
        draft_identifiers = [draft.draft_id for draft in self.drafts]
        if len(draft_identifiers) != len(set(draft_identifiers)):
            raise ValueError("Duplicate recipe-import draft IDs are not allowed.")
        question_identifiers = [question.question_id for question in self.questions]
        if len(question_identifiers) != len(set(question_identifiers)):
            raise ValueError("Duplicate recipe-import question IDs are not allowed.")
        if self.questions and not self.drafts:
            raise ValueError("Recipe-import questions require their draft snapshot.")
        return self


class ParseRecipeImportResponse(StrictModel):
    """菜谱导入解析响应。"""

    status: RecipeImportStatus
    # ↑ 解析结果状态：需要澄清 / 就绪
    drafts: tuple[RecipeImportDraft, ...]
    # ↑ 解析后的草稿集
    questions: tuple[RecipeImportQuestion, ...] = ()
    # ↑ 需要用户澄清的问题集（就绪时为空）
