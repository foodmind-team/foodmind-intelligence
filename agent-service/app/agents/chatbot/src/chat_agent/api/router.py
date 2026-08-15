"""Backend-facing chat-agent-v1 route with delegated read-only exploration."""

import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request

from chat_agent.api.backpressure import RequestLimiter
from chat_agent.api.dependencies import (
    get_backend_tool_client,
    get_llm_client,
    get_settings,
    require_internal_service,
)
from chat_agent.clients.backend import BackendToolClient, BackendToolError
from chat_agent.config.settings import Settings
from chat_agent.domain.models import AgentChatRequest, AgentChatResponse, ChatSource, GroundedSource, Route
from chat_agent.llm.client import LLMClient, LLMError

router = APIRouter(
    prefix="/internal/v1/chat",
    tags=["chat-agent"],
    dependencies=[Depends(require_internal_service)],
)
logger = logging.getLogger(__name__)

_COMPARE = re.compile(r"\b(compare|versus|vs\.?|difference)\b|比较|对比", re.IGNORECASE)
_SEARCH = re.compile(
    r"\b(find|search|look for|which|show me|list|how many|count|do you have|are there|can you see)\b"
    r"|查找|搜索|哪个|列出|有多少|多少|查看|看看|有没有",
    re.IGNORECASE,
)
_NAVIGATION = re.compile(
    r"\b(navigate|open|take me to|screen|page|how do i)\b|在哪里|页面|怎么进入|打开|带我去",
    re.IGNORECASE,
)
_COUNT = re.compile(r"\b(how many|count|number of|are there|do you have)\b|有多少|多少|数量", re.IGNORECASE)
_GREETING = re.compile(r"^(?:hi|hello|hey|你好|您好|嗨)[!！。. ]*$", re.IGNORECASE)
_PLATFORM_SCOPE = re.compile(
    r"\b(?:foodmind|platform\s+(?:item(?:s)?|feature(?:s)?|data|recommendation(?:s)?|recipe(?:s)?|record(?:s)?|"
    r"group(?:s)?|profile|preference(?:s)?)|recommend(?:ation|ations)?|meal|food|drink|recipe|cook(?:ing)?|ingredient|"
    r"restaurant(?:s)?|place(?:s)?|menu|"
    r"catalog(?:ue)?|shopping(?:\s+list)?|grocer(?:y|ies)|inventory|food\s+record|drink\s+record|history|"
    r"saved(?:\s+item|\s+recipe)?|group(?:s)?|explore|dashboard|insight(?:s)?|profile|preference(?:s)?|"
    r"allerg(?:y|en|ies)|dietary|nutrition|budget)\b"
    r"|推荐|菜谱|食谱|做饭|烹饪|餐食|食物|饮品|餐厅|菜单|目录|购物清单|库存|记录|历史|收藏|群组|"
    r"探索|仪表盘|洞察|个人资料|偏好|过敏|饮食|营养|预算",
    re.IGNORECASE,
)
_EXTERNAL_TOPIC = re.compile(
    r"\b(?:weather|forecast|temperature|rain|politics|election|stock(?:\s+market)?|crypto|sports?\s+score)\b"
    r"|天气|气温|下雨|政治|选举|股市|股票|加密货币|比赛比分",
    re.IGNORECASE,
)


@router.post("/generate", response_model=AgentChatResponse)
async def generate_chat(
    body: AgentChatRequest,
    request: Request,
    llm: Annotated[LLMClient | None, Depends(get_llm_client)],
    tools: Annotated[BackendToolClient, Depends(get_backend_tool_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentChatResponse:
    limiter: RequestLimiter = request.app.state.request_limiter
    async with limiter.lease():
        return await _generate(body, llm, tools, settings)


async def _generate(
    body: AgentChatRequest,
    llm: LLMClient | None,
    tools: BackendToolClient,
    settings: Settings,
) -> AgentChatResponse:
    if not _is_platform_related(body):
        logger.info("chat_generation outcome=unsupported reason=platform_scope trace_id=%s", body.trace_id)
        return _out_of_scope(body)

    route = _route(body)
    sources = tuple(GroundedSource.from_reference(item) for item in body.shared_references[:10])
    if route == "SEARCH":
        if not body.delegation_token:
            return _tool_unavailable(body)
        try:
            sources = await tools.search(
                query=body.message,
                delegation_token=body.delegation_token,
                timeout_seconds=_remaining_timeout(body, settings.backend_timeout_seconds),
            )
        except (BackendToolError, TimeoutError):
            return _tool_unavailable(body)
    elif route in {"SUMMARY", "COMPARE"}:
        if sources and body.delegation_token:
            try:
                resolved = await tools.resolve(
                    session_id=body.session_id,
                    reference_ids=[item.reference_id for item in body.shared_references],
                    delegation_token=body.delegation_token,
                    timeout_seconds=_remaining_timeout(body, settings.backend_timeout_seconds),
                )
                sources = resolved
            except (BackendToolError, TimeoutError):
                # Backend supplied these references immediately before the call and
                # performs a final authorisation pass over every returned source.
                pass
    else:
        sources = ()

    if llm is not None:
        try:
            answer = await llm.chat(
                _messages(body, route, sources),
                timeout_seconds=_remaining_timeout(body, settings.llm_timeout_seconds),
            )
            logger.info(
                "chat_generation outcome=success provider_host=%s model=%s route=%s trace_id=%s",
                urlsplit(settings.llm_base_url).hostname,
                settings.llm_model,
                route,
                body.trace_id,
            )
            return _response(
                body,
                route=route,
                response_status="SUCCEEDED",
                answer=_bounded(answer),
                sources=sources,
            )
        except (LLMError, TimeoutError) as exc:
            logger.warning(
                "chat_generation outcome=fallback provider_host=%s model=%s route=%s error_type=%s trace_id=%s",
                urlsplit(settings.llm_base_url).hostname,
                settings.llm_model,
                route,
                type(exc).__name__,
                body.trace_id,
            )
    else:
        logger.warning(
            "chat_generation outcome=fallback provider=disabled route=%s trace_id=%s",
            route,
            body.trace_id,
        )

    return _response(
        body,
        route=route,
        response_status="FALLBACK_SUCCEEDED",
        answer=_fallback(route, sources),
        sources=sources,
    )


def _route(body: AgentChatRequest) -> Route:
    if body.requested_route is not None:
        return body.requested_route
    if _NAVIGATION.search(body.message):
        return "NAVIGATION"
    if _COMPARE.search(body.message):
        return "COMPARE"
    if _SEARCH.search(body.message):
        return "SEARCH"
    return "SUMMARY" if body.shared_references else "NAVIGATION"


def _is_platform_related(body: AgentChatRequest) -> bool:
    """Keep chat within FoodMind instead of relying on a provider to decline unrelated prompts."""
    if _EXTERNAL_TOPIC.search(body.message):
        return False
    return bool(body.shared_references or _GREETING.fullmatch(body.message) or _PLATFORM_SCOPE.search(body.message))


def _messages(body: AgentChatRequest, route: Route, sources: tuple[GroundedSource, ...]) -> list[dict[str, str]]:
    references = [
        {
            "sourceType": item.source_type,
            "sourceId": str(item.source_id),
            "title": item.title,
            "subtitle": item.subtitle,
            "snippet": item.snippet,
        }
        for item in sources[:10]
    ]
    system = """You are FoodMind Chat, a natural, adaptable read-only assistant for the FoodMind platform.
Only answer questions about FoodMind, its features, or the user's authorised FoodMind data: meal and drink
recommendations, recipes and cooking, catalogue items and places, records, shopping lists, inventory, groups,
history, insights, profile, and preferences. Do not answer general knowledge or external topics such as weather,
news, politics, finance, or sports. If an unrelated request reaches you, say that you can only help with FoodMind
and briefly name relevant FoodMind areas. Treat supplied grounding facts as authoritative and never invent facts
attributed to FoodMind data. Never create, update, or delete data. Never write to FoodMind.
Reply in the same language as the user's message. Vary your wording and structure to fit the question.
Continue to avoid canned templates. Do not include a bibliography; FoodMind renders source cards.
Offer useful nuance, alternatives, or a brief follow-up question when that genuinely improves the answer.
Keep practical suggestions feasible and respect every stated time, ingredient, dietary, and budget constraint.
FoodMind areas: Home recommendations, Groups, Explore, Saved items, Saved recipes, Food and Drink Records,
Catalogue, Cooking Plans, Shopping Lists, Inventory, History, Insights/Dashboard, Profile, and Chat."""
    grounding_facts: dict[str, object] = {}
    if route == "SEARCH" and _COUNT.search(body.message):
        place_question = bool(
            re.search(
                r"\b(?:restaurant|restaurants|place|places)\b|餐厅|饭店|地点|场所",
                body.message,
                re.IGNORECASE,
            )
        )
        grounding_facts = {
            "verifiedCount": len(sources),
            "countIsLowerBound": any(item.grounding_metadata.get("hasNext") is True for item in sources),
            "entityLabel": "places" if place_question else "FoodMind items",
            "instruction": "State this verified count exactly; do not infer a different count.",
        }
    context = json.dumps(
        {"sources": references, "groundingFacts": grounding_facts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Selected route: {route}\nGrounded context: {context}\n\nUser message:\n{body.message}",
        },
    ]


def _fallback(route: Route, sources: tuple[GroundedSource, ...]) -> str:
    if route == "NAVIGATION":
        return (
            "I can help you navigate FoodMind. Open Home for recommendations, Groups or Explore for discovery, "
            "Saved items or Saved recipes for your shortlist, Cooking Plans, Shopping Lists or Inventory for cooking, "
            "Food and Drink Records or History for past meals, and Insights for dashboard patterns."
        )
    titles = [item.title or item.source_type.replace("_", " ").title() for item in sources[:3]]
    if route == "COMPARE":
        return f"The items available for comparison are: {', '.join(titles)}."
    if route == "SEARCH":
        return f"I found these FoodMind sources: {', '.join(titles)}."
    return f"The shared FoodMind sources are: {', '.join(titles)}."


def _out_of_scope(body: AgentChatRequest) -> AgentChatResponse:
    chinese = bool(re.search(r"[\u3400-\u9fff]", body.message))
    answer = (
        "我无法回答与 FoodMind 平台无关的问题。我可以帮助你使用 FoodMind 的推荐、食谱、记录、"
        "购物清单、库存、群组和个人偏好。"
        if chinese
        else "I can't answer questions unrelated to the FoodMind platform. I can help with FoodMind recommendations, "
        "recipes, records, shopping lists, inventory, groups, and preferences."
    )
    return _response(
        body,
        route="OUT_OF_SCOPE",
        response_status="UNSUPPORTED",
        answer=answer,
        sources=(),
    )


def _tool_unavailable(body: AgentChatRequest) -> AgentChatResponse:
    return _response(
        body,
        route="NAVIGATION",
        response_status="FALLBACK_SUCCEEDED",
        answer=("Platform search is temporarily unavailable. You can still open Records or Catalogue directly."),
        sources=(),
    )


def _remaining_timeout(body: AgentChatRequest, configured_seconds: float) -> float:
    if body.expires_at is None:
        return configured_seconds
    expiry = body.expires_at
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    remaining = (expiry - datetime.now(UTC)).total_seconds() - 0.25
    if remaining <= 0:
        raise TimeoutError
    return min(configured_seconds, remaining)


def _response(
    body: AgentChatRequest,
    *,
    route: Route,
    response_status: str,
    answer: str,
    sources: tuple[GroundedSource, ...],
) -> AgentChatResponse:
    response_sources = [
        ChatSource.model_validate(
            {
                "sourceType": item.source_type,
                "sourceId": item.source_id,
                "sequenceNo": index,
                "groundingMetadata": item.grounding_metadata,
            }
        )
        for index, item in enumerate(sources[:10], start=1)
    ]
    return AgentChatResponse.model_validate(
        {
            "requestId": body.request_id,
            "sessionId": body.session_id,
            "userMessageId": body.user_message_id,
            "traceId": body.trace_id,
            "agentTraceId": f"chat-{'llm' if response_status == 'SUCCEEDED' else 'fallback'}-{uuid.uuid4().hex}",
            "route": route,
            "responseStatus": response_status,
            "answer": _bounded(answer),
            "sources": [source.model_dump(by_alias=True) for source in response_sources],
        }
    )


def _bounded(value: str) -> str:
    cleaned = value.strip()
    return cleaned if len(cleaned) <= 4000 else cleaned[:3999].rstrip() + "…"
