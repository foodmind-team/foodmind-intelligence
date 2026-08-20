# =============================================================================
# 研究节点（workflow/research_nodes）
# -----------------------------------------------------------------------------
# 单个流水线阶段的节点实现：对低置信度关键缺口做联网研究，并把调和后的
# 证据回写候选。公共兼容面仍为 cooking_plan_agent.workflow.nodes。
# =============================================================================

"""Workflow node implementations for a single pipeline stage.

单个流水线阶段的节点实现。

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.

公共兼容面仍为 ``cooking_plan_agent.workflow.nodes``。本模块仅包含一个内聚阶段。
"""

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.models import (
    Assumption,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState


async def research_missing_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Query web research for low-confidence critical gaps (handbook 10).

    对低置信度的关键缺口查询联网研究（手册 10）。

    For each critical gap in state, query the RecipeResearcher from
    WorkflowContext. Gaps that are heat or duration related are prioritised.
    Returns evidence dict keyed by gap_id for traceability.

    对状态中的每个关键缺口，查询 WorkflowContext 中的 RecipeResearcher。
    与火候或时长相关的缺口被优先处理。返回以 gap_id 为键的证据 dict，便于追溯。

    On any failure (timeout, provider error, no results), the evidence
    stays empty and routing falls back to confirmation — never unsafe guess.

    任何失败（超时、Provider 错误、无结果）时，证据保持为空，路由回退到
    确认 —— 绝不进行不安全的猜测。
    """
    gaps = state.get("gaps", ())
    if not gaps:
        return {}

    # Only research critical gaps that are heat/duration/temperature related
    # (handbook 10.1: "only for missing cooking heat or duration")
    # 仅研究与火候 / 时长 / 温度相关的关键缺口（手册 10.1：“仅针对缺失的
    # 烹饪火候或时长”）
    _researchable_fields = {"heat_level", "duration", "temperature", "target_temperature_c"}

    researchable_gaps = [
        g
        for g in gaps
        if g.gap_class in ("critical", "safety_critical")
        and any(f in g.field_path.lower() for f in _researchable_fields)
    ]

    if not researchable_gaps:
        return {}

    # Extract dish name from recipe candidates for query context
    # 从菜谱候选中提取菜名作为查询上下文
    candidates = state.get("extracted_candidates", ())
    dish_name = candidates[0].dish_name if candidates else ""

    # Resolve each gap (handbook 10.9: at most 2 queries per dish)
    # For MVP, we use the Researcher directly rather than the Protocol
    # since the Protocol's research() signature returns list[EvidenceResult]
    # 解析每个缺口（手册 10.9：每道菜最多 2 次查询）。MVP 中我们直接使用
    # Researcher 而非 Protocol，因为 Protocol 的 research() 签名返回
    # list[EvidenceResult]
    from cooking_plan_agent.config.settings import get_settings
    from cooking_plan_agent.domain.models import ReconciledEvidence
    from cooking_plan_agent.infrastructure.cache import (
        RESEARCH_SAFETY_POLICY_VERSION,
        build_research_cache_key,
    )
    from cooking_plan_agent.research.query_builder import build_minimal_query

    settings = get_settings()
    # P1-06 cache is optional — getattr keeps duck-typed contexts working.
    # P1-06 缓存是可选的 —— getattr 使鸭子类型上下文继续可用。
    cache = getattr(runtime.context, "cache", None)

    # P1-06 research cache key: query + provider tag + safety policy version
    # (+ model for LLM-backed researchers).
    # P1-06 研究缓存键：查询 + Provider 标签 + 安全政策版本（+ 基于 LLM 的研究者的模型）。
    researcher = runtime.context.recipe_researcher
    if researcher is None:
        # The apply node still runs and supplies deterministic non-safety
        # defaults; safety gaps remain unresolved for confirmation.
        # apply 节点仍会运行并提供确定性的非安全默认值；安全缺口保持未解决以待确认。
        return {"research_evidence": {}}

    provider_tag = type(researcher).__name__
    model_tag = settings.llm_model

    async def _resolve_uncached(gap: object, query_text: str) -> ReconciledEvidence:
        resolve_gap = getattr(researcher, "resolve_gap", None)
        if callable(resolve_gap):
            try:
                resolved: ReconciledEvidence = await resolve_gap(gap, dish_name)
                return resolved
            except Exception:  # noqa: BLE001 — any failure → confirmation
                # 任何失败 → 确认
                return ReconciledEvidence(source_count=0, needs_confirmation=True)
        # Non-Researcher RecipeResearcher — Protocol research() path.
        # 非 Researcher 的 RecipeResearcher —— 走 Protocol research() 路径。
        from cooking_plan_agent.domain.models import EvidenceQuery

        query = EvidenceQuery(
            query_text=query_text,
            gap_type=gap.gap_class,  # type: ignore[attr-defined]
            recipe_context=dish_name,
        )
        try:
            results = await researcher.research(query)
            if results:
                return ReconciledEvidence(source_count=len(results), needs_confirmation=False)
            return ReconciledEvidence(source_count=0, needs_confirmation=True)
        except Exception:  # noqa: BLE001 — any failure → confirmation
            # 任何失败 → 确认
            return ReconciledEvidence(source_count=0, needs_confirmation=True)

    async def _resolve(gap: object) -> ReconciledEvidence:
        query_text = build_minimal_query(gap, dish_name)  # type: ignore[arg-type]
        if cache is None:
            return await _resolve_uncached(gap, query_text)
        key = build_research_cache_key(
            query_text,
            provider_tag=provider_tag,
            safety_policy_version=RESEARCH_SAFETY_POLICY_VERSION,
            model=model_tag,
        )
        value = await cache.get_or_compute(
            key,
            settings.cache_ttl_seconds,
            lambda: _resolve_uncached(gap, query_text),
        )
        from typing import cast

        return cast(ReconciledEvidence, value)

    max_gap_queries = getattr(researcher, "max_gap_queries", 2)
    selected_gaps = researchable_gaps if max_gap_queries is None else researchable_gaps[: max(0, int(max_gap_queries))]
    # Independent local-model inferences can run concurrently under the
    # provider client's own connection/concurrency bounds. Web researchers
    # retain the handbook limit of two queries per dish.
    # 独立的本地模型推断可在 Provider 客户端自身的连接 / 并发限制下并发运行。
    # 联网研究者保留手册“每道菜最多两次查询”的限制。
    import asyncio

    async def _resolve_bounded(gap: object) -> ReconciledEvidence:
        try:
            return await asyncio.wait_for(
                _resolve(gap),
                timeout=settings.research_timeout_seconds,
            )
        except TimeoutError:
            return ReconciledEvidence(source_count=0, needs_confirmation=True)

    resolved_evidence = await asyncio.gather(*(_resolve_bounded(gap) for gap in selected_gaps))
    research_evidence = {gap.gap_id: resolved for gap, resolved in zip(selected_gaps, resolved_evidence, strict=True)}

    return {"research_evidence": research_evidence}


async def apply_research_evidence_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Write reconciled research evidence back into candidates (P1-01).

    把调和后的研究证据回写进候选（P1-01）。

    Sits between ``research_missing`` and IR validation so search results
    actually update the plan — research is no longer a write-only bypass.

    位于 ``research_missing`` 与 IR 验证之间，使搜索结果真正更新计划 ——
    研究不再是只写不用的旁路。

    For each gap with evidence:
      1. Locates the exact target via ``gap_id + recipe_id + field_path``
         (never by list position).
      2. Applies reliable values (heat / duration / temperature) to the
         candidate step and records an EvidenceRef-backed Assumption.
      3. Marks the gap resolved; only unresolved gaps stay in state.

    对每个有证据的缺口：
      1. 通过 ``gap_id + recipe_id + field_path`` 定位确切目标（绝不按列表位置）。
      2. 将可靠值（火候 / 时长 / 温度）应用到候选步骤，并记录一条由
         EvidenceRef 支撑的 Assumption。
      3. 标记缺口已解决；只有未解决的缺口保留在状态中。

    Anything that cannot be safely auto-applied — no source, disagreement,
    field-location failure, or a safety-critical temperature without a
    verifiable URL — sets ``needs_confirmation`` so routing surfaces the
    user confirmation instead of silently guessing (P1-01 rules 5 & 6).

    任何无法安全自动应用的情况 —— 无来源、分歧、字段定位失败、或没有
    可验证 URL 的安全关键温度 —— 都会设置 ``needs_confirmation``，使路由
    暴露用户确认而非静默猜测（P1-01 规则 5 & 6）。
    """
    from cooking_plan_agent.research.evidence_apply import apply_evidence_to_candidate

    research_evidence = state.get("research_evidence", {})

    candidates = list(state.get("extracted_candidates", ()))
    gaps = state.get("gaps", ())

    applied_gap_ids: set[str] = set()
    assumptions: list[Assumption] = []
    needs_confirmation = False

    for gap in gaps:
        # Locate the recipe by stable recipe_id, never by list position.
        # 通过稳定的 recipe_id 定位菜谱，绝不按列表位置。
        candidate_idx = next(
            (i for i, candidate in enumerate(candidates) if candidate.recipe_id == gap.recipe_id),
            None,
        )
        if candidate_idx is None:
            needs_confirmation = True  # recipe-level location failure  # 菜谱级定位失败
            continue

        reconciled = research_evidence.get(gap.gap_id)
        if reconciled is None or reconciled.source_count <= 0:
            from cooking_plan_agent.parsing.inference import (
                InferenceResult,
                infer_deterministic_default,
                merge_inference,
            )

            fallback = infer_deterministic_default(candidates[candidate_idx], gap)
            if fallback is not None:
                filled_gap, assumption = fallback
                candidates[candidate_idx] = merge_inference(
                    candidates[candidate_idx],
                    InferenceResult((filled_gap,), (), (assumption,)),
                )
                applied_gap_ids.add(gap.gap_id)
                assumptions.append(assumption)
                continue
            # Safety and non-operational gaps stay unresolved.
            # 安全与非操作性缺口保持未解决。
            continue

        result = apply_evidence_to_candidate(candidates[candidate_idx], gap, reconciled)
        if result.applied and result.candidate is not None:
            candidates[candidate_idx] = result.candidate
            applied_gap_ids.add(gap.gap_id)
            if result.assumption is not None:
                assumptions.append(result.assumption)
            # Even applied values that came from conflicting evidence (MAD
            # over threshold) must surface for user confirmation — never
            # silently adopt a disputed value (P1-01 rule 5).
            # 即使已应用的值来自冲突证据（MAD 超过阈值），也必须暴露给用户
            # 确认 —— 绝不静默采纳有争议的值（P1-01 规则 5）。
            if reconciled.needs_confirmation:
                needs_confirmation = True
        else:
            from cooking_plan_agent.parsing.inference import (
                InferenceResult,
                infer_deterministic_default,
                merge_inference,
            )

            fallback = infer_deterministic_default(candidates[candidate_idx], gap)
            if fallback is not None:
                filled_gap, assumption = fallback
                candidates[candidate_idx] = merge_inference(
                    candidates[candidate_idx],
                    InferenceResult((filled_gap,), (), (assumption,)),
                )
                applied_gap_ids.add(gap.gap_id)
                assumptions.append(assumption)
            else:
                needs_confirmation = needs_confirmation or result.needs_confirmation

    remaining_gaps = tuple(g for g in gaps if g.gap_id not in applied_gap_ids)
    if any(g.gap_class in ("critical", "safety_critical") for g in remaining_gaps):
        needs_confirmation = True

    return {
        "extracted_candidates": tuple(candidates),
        "gaps": remaining_gaps,
        "research_assumptions": tuple(assumptions),
        "needs_confirmation": needs_confirmation,
    }
