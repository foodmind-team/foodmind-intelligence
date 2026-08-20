"""Spring Boot v1 compat router — the endpoint the Java caller actually hits.

POST /internal/v1/cooking-plans/generate

This router mirrors the native agent endpoint but speaks the Java
``cooking-agent-v1`` contract (Bearer auth, camelCase DTOs, deadline
budget).  It is the P0-02 integration baseline: the Spring Boot caller
is unchanged; this endpoint simply translates.

Native endpoint (unchanged): POST /internal/v1/agents/cooking-plan/generate
"""

# 模块概览（中文）：这是 Java 调用方实际命中的兼容路由。
# 端点：POST /internal/v1/cooking-plans/generate
# 作用：镜像原生 Agent 端点，但改用 Java 的 cooking-agent-v1 契约（Bearer 鉴权、
#       camelCase DTO、deadline 预算）。它是 P0-02 的集成基线——Java 调用方无需改动，
#       本端点只做“翻译/适配”。
# 注意：原生端点（不变）是 POST /internal/v1/agents/cooking-plan/generate。

from __future__ import annotations

import logging
from datetime import UTC, datetime  # UTC：固定时区；datetime：解析/比较 deadline
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cooking_plan_agent.api.backpressure import request_lease  # 请求级背压（P1-03）
from cooking_plan_agent.api.compat_models import (
    CompatCookingRequest,  # 入参：完整请求体
    CompatCookingResponse,  # 出参：完整响应体
)
from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,  # 提取关联 ID（链路追踪）
    require_bearer_service,  # Bearer 鉴权依赖
)
from cooking_plan_agent.application import GenerateCookingPlanService  # 应用服务
from cooking_plan_agent.application.contract_adapter import (
    build_internal_request,  # 把兼容请求映射为内部请求
    deadline_budget_seconds,  # 计算 deadline 剩余预算（秒）
    is_contract_supported,  # 判断契约版本是否受支持
    selected_recipe_id,  # 取最终选中的菜谱 ID
    to_compat_response,  # 把内部结果映射为兼容响应
)
from cooking_plan_agent.infrastructure.checkpointer import build_thread_id  # 构造检查点 thread_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/v1/cooking-plans",
    tags=["cooking-plan-agent-v1-compat"],
    dependencies=[Depends(require_bearer_service)],  # 路由级 Bearer 鉴权（对所有子路由生效）
)


def get_generate_service(request: Request) -> GenerateCookingPlanService:
    """Retrieve the application service from the app's lifespan-injected state."""
    # 从应用启动（lifespan）注入的 state 中取出应用服务
    service = request.app.state.generate_plan_service
    if not isinstance(service, GenerateCookingPlanService):
        # 未初始化则快速失败，而非返回一个错误类型
        raise AttributeError("generate_plan_service was not initialised during startup")
    return service


@router.post("/generate", response_model=CompatCookingResponse)
async def generate_plan_compat(
    body: CompatCookingRequest,  # 请求体（Pydantic 已按 CompatCookingRequest 校验/反序列化）
    service: Annotated[GenerateCookingPlanService, Depends(get_generate_service)],
    _correlation_id: Annotated[str, Depends(extract_correlation_id)],  # 关联 ID（仅用于链路追踪）
    _lease: Annotated[None, Depends(request_lease)] = None,  # 背压租约（前置获取，后置释放）
) -> CompatCookingResponse:
    """Generate a cooking plan for the Spring Boot v1 contract.

    - Validates the contract version (fast fail on unsupported version).
    - Enforces the request's deadlineAt as an execution budget (fast fail
      when the deadline has already passed).
    - Maps candidate snapshots to internal structured candidates so no LLM
      call is made (P0-02 rule 4).
    - READY → SUCCEEDED; any other terminal state → FAILED.
    """
    # ① 校验契约版本：不支持的版本直接失败（快速失败）
    if not is_contract_supported(body.contractVersion):
        logger.warning(
            "Unsupported contract version | contract_version=%s",
            body.contractVersion,
        )
        return _failure_response(body, "UNSUPPORTED_CONTRACT_VERSION")

    # ② Deadline 预算：调用方 deadline 已过期则快速失败
    now = datetime.now(UTC)
    budget = deadline_budget_seconds(body.deadlineAt, now)
    if budget is not None and budget <= 0:
        logger.info("Request deadline already passed | request_id=%s", body.requestId)
        return _failure_response(body, "DEADLINE_PASSED")

    # ③ 把兼容请求映射为内部结构化请求（候选方案直接映射，不调用 LLM，见 P0-02 规则 4）
    internal_request = build_internal_request(body)
    if not internal_request.preparsed_candidates:
        # 没有可用候选 → Java 侧会映射为 NO_RECIPE_MATCH
        return _failure_response(body, "NO_USABLE_CANDIDATES")

    # ④ 执行应用服务
    response = await service.execute(
        internal_request,
        # P5-0: 与原生路由保持一致——按“每次请求尝试”命名空间隔离检查点状态，
        # 使得兼容端点在启用检查点持久化（P2-06）时也能正常工作。
        thread_id=build_thread_id(str(body.requestId)),
    )
    source_recipe_id = selected_recipe_id(body)  # 取最终选中的菜谱 ID
    return to_compat_response(body, response, source_recipe_id)


def _failure_response(
    body: CompatCookingRequest,
    code: str,
) -> CompatCookingResponse:
    """Build a FAILED compat response with the envelope echoed back."""
    # 构造 FAILED 兼容响应：回显请求信封（requestId/planId/traceId 等），状态置为 FAILED
    from uuid import uuid4  # 延迟导入，仅为生成 agentTraceId

    return CompatCookingResponse(
        contractVersion=body.contractVersion,
        requestId=body.requestId,
        planId=body.planId,
        traceId=body.traceId,
        agentTraceId=uuid4().hex,  # Agent 侧生成一个随机追踪 ID
        status="FAILED",
        servings=body.request.servings,
        estimatedCost=None,
        currency=body.request.currency,
    )
