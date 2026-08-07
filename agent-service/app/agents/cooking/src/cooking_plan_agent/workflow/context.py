"""WorkflowContext — dependency injection container for LangGraph nodes.

Per handbook 8.3: services are passed through runtime context, keeping
state serialisable even without checkpoint persistence in MVP.
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
# ---------------------------------------------------------------------------
# Protocols are used instead of ABCs so that ANY object satisfying the
# shape (duck typing) can be injected — no explicit inheritance needed.
# This keeps domain services decoupled from the workflow layer.


@runtime_checkable
class RecipeExtractor(Protocol):
    """Parse unstructured recipe text into ExtractedRecipeCandidate.

    May use LLM, regex, or both — the workflow does not care about the
    implementation, only the async extract() contract.
    """

    async def extract(self, source_text: str) -> "ExtractedRecipeCandidate": ...


@runtime_checkable
class RecipeResearcher(Protocol):
    """Search for evidence to fill recipe gaps.

    Returns a list because a single query may yield multiple evidence
    sources (e.g., temperatures from different databases).
    """

    async def research(self, query: "EvidenceQuery") -> list["EvidenceResult"]: ...


@runtime_checkable
class SafetyRuleEngine(Protocol):
    """Evaluate food safety constraints against parsed recipes.

    Returns a SafetyReport aggregating all rule findings. The report
    drives routing decisions (is_safe → proceed; has_unrepairable → INFEASIBLE).
    """

    def evaluate(self, context: "SafetyContext") -> "SafetyReport": ...


@runtime_checkable
class PlanExplainer(Protocol):
    """Explain a solved schedule in natural language (P4-01).

    Receives a compact, NON-SENSITIVE summary (makespan, dish completions,
    parallel groups) — never raw recipes, inventory, or user identity (D4).
    Returns prose; the caller must treat the explanation as additive so a
    failure never blocks the READY response.
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


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable dependency context for all workflow nodes.

    Created at app startup (FastAPI lifespan). Each node receives this
    via LangGraph runtime.context. All services are swappable for testing.

    Frozen=True ensures nodes cannot mutate shared state through the context
    — all state changes must flow through PlanState returns.
    """

    recipe_extractor: RecipeExtractor
    # recipe_researcher is None in MVP; when wired, it enables the
    # research_missing node to fill low-confidence critical gaps
    recipe_researcher: RecipeResearcher | None = None
    # safety_engine evaluates food safety constraints before scheduling.
    # When None (backwards-compat), validate_safety_node returns a safe stub.
    safety_engine: SafetyRuleEngine | None = None
    # P1-06: intermediate-artifact cache (parse/research results). None keeps
    # the pipeline fully uncached — results are identical either way.
    cache: "Cache | None" = None
    # P4-01: optional schedule explainer. When None (or disabled via
    # Settings.explanation_enabled) the explain node emits no explanation or
    # a deterministic one — the READY response is never blocked.
    explainer: PlanExplainer | None = None
    # P5-3: 可选 LLM 诊断器（加法能力，缺失则纯规则修复）。
    repair_diagnoser: "RepairDiagnoser | None" = None
    # P5-2: 可选 ReAct 控制器。缺失或未启用时图直接走确定性 DAG。
    agent_controller: "AgentController | None" = None
    # P5-4: 可选长期偏好存储。缺失或请求无 user_id 时不注入记忆（零回归）。
    preference_store: "object | None" = None
    # Future services (all optional until fully implemented):
    # optimiser: ScheduleOptimiser | None = None
