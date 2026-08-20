# =============================================================================
# 工作流上下文模块（workflow/context）
# -----------------------------------------------------------------------------
# LangGraph 节点的依赖注入容器。按手册 8.3：服务通过运行时上下文传递，
# 使状态即使在 MVP 无检查点持久化时也保持可序列化。
# =============================================================================

"""WorkflowContext — dependency injection container for LangGraph nodes.

WorkflowContext —— LangGraph 节点的依赖注入容器。

Per handbook 8.3: services are passed through runtime context, keeping
state serialisable even without checkpoint persistence in MVP.

按手册 8.3：服务通过运行时上下文传递，使状态即使在 MVP 无检查点持久化时也保持可序列化。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cooking_plan_agent.domain.models import (
        EvidenceQuery,
        EvidenceResult,
        ExtractedRecipeCandidate,
        SafetyContext,
        SafetyReport,
    )
    from cooking_plan_agent.infrastructure.cache import Cache


# ---------------------------------------------------------------------------
# Service protocols for the workflow context
# 工作流上下文的服务协议
# ---------------------------------------------------------------------------
# Protocols are used instead of ABCs so that ANY object satisfying the
# shape (duck typing) can be injected — no explicit inheritance needed.
# This keeps domain services decoupled from the workflow layer.
# 使用协议而非抽象基类（ABC），使任何满足该形状（鸭子类型）的对象都可被注入
# —— 无需显式继承。这让领域服务与工作流层解耦。


@runtime_checkable
class RecipeExtractor(Protocol):
    """Parse unstructured recipe text into ExtractedRecipeCandidate.

    将非结构化菜谱文本解析为 ExtractedRecipeCandidate。

    May use LLM, regex, or both — the workflow does not care about the
    implementation, only the async extract() contract.

    可使用 LLM、正则或两者 —— 工作流不关心实现，只关心 async extract() 契约。
    """

    async def extract(self, source_text: str) -> "ExtractedRecipeCandidate": ...


@runtime_checkable
class RecipeResearcher(Protocol):
    """Search for evidence to fill recipe gaps.

    搜索证据以填补菜谱缺口。

    Returns a list because a single query may yield multiple evidence
    sources (e.g., temperatures from different databases).

    返回列表，因为单个查询可能产出多个证据来源（如不同数据库的温度）。
    """

    async def research(self, query: "EvidenceQuery") -> list["EvidenceResult"]: ...


@runtime_checkable
class SafetyRuleEngine(Protocol):
    """Evaluate food safety constraints against parsed recipes.

    对已解析的菜谱评估食品安全约束。

    Returns a SafetyReport aggregating all rule findings. The report
    drives routing decisions (is_safe → proceed; has_unrepairable → INFEASIBLE).

    返回聚合所有规则发现的 SafetyReport。该报告驱动路由决策
    （is_safe → 继续；has_unrepairable → INFEASIBLE）。
    """

    def evaluate(self, context: "SafetyContext") -> "SafetyReport": ...


@runtime_checkable
class PlanExplainer(Protocol):
    """Explain a solved schedule in natural language (P4-01).

    用自然语言解释已求解的排程（P4-01）。

    Receives a compact, NON-SENSITIVE summary (makespan, dish completions,
    parallel groups) — never raw recipes, inventory, or user identity (D4).
    Returns prose; the caller must treat the explanation as additive so a
    failure never blocks the READY response.

    接收紧凑、非敏感的摘要（总时长、各菜完成时间、并行组）—— 绝不接收
    原始菜谱、库存或用户身份（D4）。返回散文；调用方必须将解释视为加法能力，
    使失败绝不阻塞 READY 响应。
    """

    async def explain(self, schedule_summary: dict[str, Any]) -> str: ...


@runtime_checkable
class RepairDiagnoser(Protocol):
    """P5-3: 可选 LLM 诊断器 —— 对验证失败做摘要。

    只输入 issue code 列表与求解深度（非敏感），返回建议 dict。
    建议不直接产生副作用：最终动作由 repair 规则裁决。
    """

    async def diagnose(self, context: dict[str, object]) -> dict[str, object]: ...


@runtime_checkable
class AgentController(Protocol):
    """P5-2: LLM 控制器 —— 决定下一步动作。

    Returns 结构化决策：
      {"type": "tool_call", "tool": str, "arguments": dict} |
      {"type": "final", "response": dict} |
      {"type": "fallback"}   # 控制器主动放弃，交回确定性 DAG
    """

    async def decide(self, state_summary: dict[str, object]) -> dict[str, object]: ...


@runtime_checkable
class PreferenceStore(Protocol):
    """P5-4: 长期偏好存储 —— 读/写用户已确认的偏好。

    仅存储用户显式提供/确认过的信息（隐私：不记录原始菜谱文本）。
    ``get`` 对未知 user_id 返回空 dict，保证无记忆时为「零操作」。
    """

    def get(self, user_id: str) -> dict[str, object]: ...

    def put(self, user_id: str, payload: dict[str, object]) -> None: ...


# ---------------------------------------------------------------------------
# Context dataclass
# 上下文 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable dependency context for all workflow nodes.

    所有工作流节点的不可变依赖上下文。

    Created at app startup (FastAPI lifespan). Each node receives this
    via LangGraph runtime.context. All services are swappable for testing.

    在应用启动时创建（FastAPI lifespan）。每个节点通过 LangGraph runtime.context
    接收它。所有服务都可替换以进行测试。

    Frozen=True ensures nodes cannot mutate shared state through the context
    — all state changes must flow through PlanState returns.

    Frozen=True 确保节点无法通过上下文修改共享状态 —— 所有状态变更必须
    通过 PlanState 返回值流转。
    """

    recipe_extractor: RecipeExtractor
    # recipe_researcher is None in MVP; when wired, it enables the
    # research_missing node to fill low-confidence critical gaps
    # recipe_researcher 在 MVP 中为 None；接线后，它使 research_missing
    # 节点能够填补低置信度的关键缺口
    recipe_researcher: RecipeResearcher | None = None
    # safety_engine evaluates food safety constraints before scheduling.
    # When None (backwards-compat), validate_safety_node returns a safe stub.
    # safety_engine 在排程前评估食品安全约束。为 None（向后兼容）时，
    # validate_safety_node 返回安全存根。
    safety_engine: SafetyRuleEngine | None = None
    # P1-06: intermediate-artifact cache (parse/research results). None keeps
    # the pipeline fully uncached — results are identical either way.
    # P1-06：中间产物缓存（解析 / 研究结果）。None 时管线完全不缓存 —— 结果相同。
    cache: "Cache | None" = None
    # P4-01: optional schedule explainer. When None (or disabled via
    # Settings.explanation_enabled) the explain node emits no explanation or
    # a deterministic one — the READY response is never blocked.
    # P4-01：可选排程解释器。为 None（或通过 Settings.explanation_enabled 禁用）时，
    # 解释节点不输出解释或输出确定性解释 —— READY 响应绝不被阻塞。
    explainer: PlanExplainer | None = None
    # P5-3: 可选 LLM 诊断器（加法能力，缺失则纯规则修复）。
    repair_diagnoser: "RepairDiagnoser | None" = None
    # P5-2: 可选 ReAct 控制器。缺失或未启用时图直接走确定性 DAG。
    agent_controller: "AgentController | None" = None
    # P5-4: 可选长期偏好存储。缺失或请求无 user_id 时不注入记忆（零回归）。
    preference_store: PreferenceStore | None = None
