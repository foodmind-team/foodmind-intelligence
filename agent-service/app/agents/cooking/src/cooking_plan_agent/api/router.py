"""Internal API router for the cooking plan agent.

Per handbook 9.2: one endpoint for generation.
POST /internal/v1/agents/cooking-plan/generate

The Spring Boot caller converts pasted text or .txt upload into recipe
text before calling this endpoint. This keeps the internal contract
JSON-only and prevents duplicated multipart/file rules.

Handbook 9.1: the public boundary stays in Spring Boot — this router
validates service authentication and the internal request schema, not
end-user JWTs.
"""

# 模块概览（中文）：烹饪计划 Agent 的内部 API 路由。
# 端点（Handbook 9.2）：POST /internal/v1/agents/cooking-plan/generate
# 边界：粘贴文本/上传 .txt 由 Spring Boot 调用方先转成菜谱文本再调本端点，
#       使内部契约保持纯 JSON，避免重复的 multipart/文件规则。
# Handbook 9.1：公共边界在 Spring Boot 侧——本路由只校验服务鉴权与内部请求 schema，不校验终端用户 JWT。

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from cooking_plan_agent.api.backpressure import request_lease  # 请求级背压（P1-03）
from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,  # 提取关联 ID
    require_internal_service,  # 内部服务鉴权（X-Internal-Token）
)
from cooking_plan_agent.application import GenerateCookingPlanService, ParseRecipeImportService
from cooking_plan_agent.application.recipe_import_service import InvalidRecipeImportAnswers
from cooking_plan_agent.domain.models import (
    ConfirmationAnswersRequest,
    ErrorEnvelope,
    GeneratePlanRequest,
    PlanResponse,
    PreprocessRecipesRequest,
    PreprocessRecipesResponse,
)
from cooking_plan_agent.domain.recipe_imports import ParseRecipeImportRequest, ParseRecipeImportResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router — all endpoints require internal service authentication
# 路由——所有端点都要求内部服务鉴权
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/internal/v1/agents/cooking-plan",
    tags=["cooking-plan-agent"],
    dependencies=[Depends(require_internal_service)],  # 路由级内部服务鉴权
)


# ---------------------------------------------------------------------------
# Service dependency — extracted from request.app.state
# 服务依赖——从 request.app.state 提取
# ---------------------------------------------------------------------------


def get_generate_service(request: Request) -> GenerateCookingPlanService:
    """Retrieve the application service from the app's lifespan-injected state.

    Raises AttributeError if the service was not initialised during startup.
    """
    # 从启动时注入的 state 取应用服务；未初始化则抛 AttributeError
    service = request.app.state.generate_plan_service
    if not isinstance(service, GenerateCookingPlanService):
        raise AttributeError("generate_plan_service was not initialised during startup")
    return service


def get_recipe_import_service(request: Request) -> ParseRecipeImportService:
    """Retrieve the recipe-import use case from lifespan-managed state."""
    # 从 lifespan 管理的 state 取菜谱导入用例服务
    service = request.app.state.recipe_import_service
    if not isinstance(service, ParseRecipeImportService):
        raise AttributeError("recipe_import_service was not initialised during startup")
    return service


@router.post(
    "/recipe-imports/parse",
    response_model=ParseRecipeImportResponse,
    responses={
        401: {"model": ErrorEnvelope, "description": "Authentication failed."},
        422: {"model": ErrorEnvelope, "description": "Input or answers are invalid."},
        503: {"model": ErrorEnvelope, "description": "Overloaded or shutting down."},
    },
)
async def parse_recipe_import(
    body: ParseRecipeImportRequest,
    service: Annotated[ParseRecipeImportService, Depends(get_recipe_import_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
    _lease: Annotated[None, Depends(request_lease)] = None,  # 背压租约
) -> ParseRecipeImportResponse:
    """Parse multilingual recipe text into English drafts and structured follow-ups."""
    # 解析多语言菜谱文本为英文草稿 + 结构化追问
    try:
        return await service.execute(body)
    except InvalidRecipeImportAnswers as exception:
        # 追问答案非法 → 422
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "INVALID_RECIPE_IMPORT_ANSWERS", "message": str(exception)},
        ) from exception


# ---------------------------------------------------------------------------
# Preprocess endpoint — NL parsing + gap filling, reused by the backend
# 预处理端点——自然语言解析 + 缺口填补，被后端复用
# ---------------------------------------------------------------------------


@router.post(
    "/preprocess",
    response_model=PreprocessRecipesResponse,
    responses={
        401: {"model": ErrorEnvelope, "description": "Authentication failed."},
        422: {"model": ErrorEnvelope, "description": "Request validation failed."},
        503: {"model": ErrorEnvelope, "description": "Overloaded or shutting down."},
        500: {"model": ErrorEnvelope, "description": "Unexpected internal error."},
    },
)
async def preprocess_recipes(
    body: PreprocessRecipesRequest,
    service: Annotated[GenerateCookingPlanService, Depends(get_generate_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
    _lease: Annotated[None, Depends(request_lease)] = None,  # 背压租约
) -> PreprocessRecipesResponse:
    """Parse raw recipe text and fill missing fields (NL + gap pipeline).

    The Spring Boot backend calls this endpoint BEFORE generate() so it can
    reuse the agent's recipe-understanding pipeline: raw text in, fully
    populated ``ExtractedRecipeCandidate`` out. The backend passes those
    candidates back on the generate request as ``preparsed_candidates``,
    so generate never re-parses and never asks gap/assumption questions —
    the agent stays focused on planning while its parsing capability is
    reused by the backend.
    """
    # 解析原始菜谱文本并填补缺失字段（NL + 缺口流水线）。
    # Spring Boot 在 generate() 之前调用本端点，复用 Agent 的菜谱理解能力：
    # 输入原始文本，输出完整填充的 ExtractedRecipeCandidate。后端再把这些候选
    # 作为 preparsed_candidates 传回 generate，使 generate 不再重复解析、不再追问缺口问题。
    logger.info(
        "Preprocessing recipes | request_id=%s | recipes=%d",
        body.request_id,
        len(body.recipes),
    )
    return await service.preprocess(body)


# ---------------------------------------------------------------------------
# Generate endpoint
# 生成端点
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=PlanResponse,
    responses={
        401: {"model": ErrorEnvelope, "description": "Authentication failed."},
        422: {"model": ErrorEnvelope, "description": "Request validation failed."},
        503: {"model": ErrorEnvelope, "description": "Overloaded or shutting down."},
        500: {"model": ErrorEnvelope, "description": "Unexpected internal error."},
    },
)
async def generate_plan(
    body: GeneratePlanRequest,
    service: Annotated[GenerateCookingPlanService, Depends(get_generate_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
    _lease: Annotated[None, Depends(request_lease)] = None,  # 背压租约（P1-03）
) -> PlanResponse:
    """Generate a cooking plan from the supplied recipes and constraints.

    Accepts a JSON request body validated against GeneratePlanRequest.
    Returns a PlanResponse — one of READY, NEEDS_CONFIRMATION, INFEASIBLE,
    or FAILED. All business outcomes return HTTP 200 per handbook 9.8.

    The correlation ID is injected via the X-Request-ID dependency and
    propagated to response headers by the CORSMiddleware (configured in
    create_app).

    P1-03: the request_lease dependency bounds concurrency — when the
    active/queued limiter is saturated the request is rejected with 503 +
    Retry-After instead of piling onto the event loop.
    """
    # 生成烹饪计划：返回 PlanResponse（READY / NEEDS_CONFIRMATION / INFEASIBLE / FAILED），
    # 所有业务结果均返回 HTTP 200（handbook 9.8）。
    logger.info(
        "Generating plan | request_id=%s | recipes=%d | time_limit=%s",
        body.request_id,
        len(body.recipes),
        body.time_limit_minutes,
    )
    # P2-06: thread_id namespaces checkpoint state per request attempt
    # (request_id + plan_revision), enabling resume after process restart.
    # P2-06：thread_id 按“每次请求尝试”命名空间隔离检查点状态
    # （request_id + plan_revision），支持进程重启后恢复。
    from cooking_plan_agent.infrastructure.checkpointer import build_thread_id

    thread_id = build_thread_id(body.request_id, body.plan_revision)
    return await service.execute(body, thread_id=thread_id)


# ---------------------------------------------------------------------------
# P5-4: Confirm endpoint — resume a paused NEEDS_CONFIRMATION dialog
# P5-4：确认端点——恢复暂停的 NEEDS_CONFIRMATION 对话
# ---------------------------------------------------------------------------


@router.post(
    "/plans/{plan_id}/confirm",
    response_model=PlanResponse,
    responses={
        401: {"model": ErrorEnvelope, "description": "Authentication failed."},
        422: {"model": ErrorEnvelope, "description": "Request validation failed."},
        503: {"model": ErrorEnvelope, "description": "Overloaded or shutting down."},
    },
)
async def confirm_plan(
    plan_id: str,
    body: ConfirmationAnswersRequest,
    service: Annotated[GenerateCookingPlanService, Depends(get_generate_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],
) -> PlanResponse:
    """Resume a NEEDS_CONFIRMATION plan with the user's answers (P5-4).

    Accepts the ConfirmationQuestion answers submitted against the
    confirmation form returned by a prior generation (``plan_revision``
    echoes the confirmation the client is answering). The answers resume
    the same checkpoint thread, so the dialog continues toward READY or
    another confirmation turn.

    Requires ``confirmation_dialog_enabled=true`` and an active
    checkpointer; otherwise there is no paused dialog to resume and the
    service returns FAILED.
    """
    # 用用户答案恢复一个 NEEDS_CONFIRMATION 的计划（P5-4）。
    # 要求 confirmation_dialog_enabled=true 且有活跃 checkpointer；否则无可恢复的暂停对话，服务返回 FAILED。
    logger.info(
        "Resuming confirmation | plan_id=%s | answers=%d | revision=%s",
        plan_id,
        len(body.answers),
        body.plan_revision,
    )
    from cooking_plan_agent.infrastructure.checkpointer import build_thread_id

    thread_id = build_thread_id(body.plan_id, body.plan_revision)
    return await service.continue_after_confirmation(body, thread_id=thread_id)
