# =============================================================================
# 输入解析与验证节点（workflow/input_parsing）
# -----------------------------------------------------------------------------
# 单个流水线阶段的节点实现：请求校验、菜谱解析、缺口检测、本地推断与
# RecipeIR 验证。公共兼容面仍为 cooking_plan_agent.workflow.nodes。
# =============================================================================

"""Workflow node implementations for a single pipeline stage.

单个流水线阶段的节点实现。

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.

公共兼容面仍为 ``cooking_plan_agent.workflow.nodes``。本模块仅包含一个内聚阶段。
"""

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    Assumption,
    ExtractedRecipeCandidate,
    GeneratePlanRequest,
    RecipeGap,
    RecipeIR,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# P5-4: 长期偏好字段 —— 仅这些字段从记忆注入，且显式请求值优先。
_PREFERENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("dietary_restrictions", "dietary_restrictions"),
    ("user_allergens", "allergens"),
)


def merge_preferences_into_request(
    request: GeneratePlanRequest | dict[str, object],
    preferences: dict[str, object],
) -> dict[str, object]:
    """把用户已确认的长期偏好合并进请求（P5-4）。

    合并规则：显式请求值 > 记忆值。用户当场提交的字段（即使为空元组）
    视为显式值，覆盖记忆中的对应字段；仅当请求未携带该字段时才注入
    记忆值。无 user_id / 无记忆时为空操作（零回归）。
    """
    if not preferences:
        if isinstance(request, GeneratePlanRequest):
            return request.model_dump()
        return dict(request)

    merged = request.model_dump() if isinstance(request, GeneratePlanRequest) else dict(request)
    for request_field, memory_field in _PREFERENCE_FIELDS:
        # 请求中显式声明过（含空元组）则不注入记忆值。
        if request_field in merged and merged[request_field] is not None:
            continue
        memory_value = preferences.get(memory_field)
        if isinstance(memory_value, (list, tuple)):
            merged[request_field] = tuple(memory_value)
    return merged


async def validate_input_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Validate the incoming GeneratePlanRequest (P0-03, P0-05).

    校验传入的 GeneratePlanRequest（P0-03、P0-05）。

    Checks (in order so the FIRST violation wins):
      1. Non-empty recipes
      2. recipe_id uniqueness
      3. Recipe count vs Settings.max_recipe_count
      4. Per-recipe text byte size vs Settings.max_recipe_text_bytes
      5. Total serialised request size vs Settings.max_request_bytes
      6. schema_version in Settings.supported_schema_versions
      7. time_limit_minutes is non-negative

    检查（按顺序，首个违规者胜出）：
      1. 菜谱非空
      2. recipe_id 唯一性
      3. 菜谱数量 vs Settings.max_recipe_count
      4. 每菜谱文本字节大小 vs Settings.max_recipe_text_bytes
      5. 序列化请求总大小 vs Settings.max_request_bytes
      6. schema_version 在 Settings.supported_schema_versions 中
      7. time_limit_minutes 非负

    Returns an empty dict on success or {'error': WorkflowError} on failure.
    The graph routes any error straight to the FAILED terminal.

    成功时返回空 dict，失败时返回 {'error': WorkflowError}。
    图将任何错误直接路由到 FAILED 终态。
    """
    from cooking_plan_agent.config.settings import get_settings

    request = state["request"]
    settings = get_settings()

    # --- P5-4: 长期偏好注入（仅当 context 提供 PreferenceStore 且请求带 user_id）。
    # 显式请求值优先；无 user_id / 无记忆时为零操作（零回归）。
    preference_store = getattr(runtime.context, "preference_store", None)
    if preference_store is not None and request.user_id:
        preferences = preference_store.get(request.user_id)
        if preferences:
            merged = merge_preferences_into_request(request, preferences)
            request = GeneratePlanRequest.model_validate(merged)

    # 1. Non-empty recipes — keep the original request error code (P0-03:
    #    must not be overwritten by downstream SCHEDULE_INFEASIBLE).
    # 1. 菜谱非空 —— 保留原始请求错误码（P0-03：绝不能被下游的
    #    SCHEDULE_INFEASIBLE 覆盖）。
    if not request.recipes:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                message="Request contains no recipes",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 2. Duplicate recipe IDs  2. 重复的菜谱 ID
    seen: set[str] = set()
    for recipe in request.recipes:
        if recipe.recipe_id in seen:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.DUPLICATE_RECIPE_ID.value,
                    message=f"Duplicate recipe_id: {recipe.recipe_id}",
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }
        seen.add(recipe.recipe_id)

    # 3. Recipe count cap  3. 菜谱数量上限
    if len(request.recipes) > settings.max_recipe_count:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.TOO_MANY_RECIPES.value,
                message=(
                    f"Request contains {len(request.recipes)} recipes, max allowed is {settings.max_recipe_count}"
                ),
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 4. Per-recipe text byte cap  4. 每菜谱文本字节上限
    for recipe in request.recipes:
        text_bytes = len(recipe.text.encode("utf-8"))
        if text_bytes > settings.max_recipe_text_bytes:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.RECIPE_TEXT_TOO_LARGE.value,
                    message=(
                        f"Recipe '{recipe.recipe_id}' text is {text_bytes} bytes, "
                        f"max allowed is {settings.max_recipe_text_bytes}"
                    ),
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

    # 5. Total serialised request size  5. 序列化请求总大小
    try:
        total_bytes = len(request.model_dump_json().encode("utf-8"))
    except Exception:  # noqa: BLE001 — serialisation must not crash validation
        # 序列化绝不能导致校验崩溃
        total_bytes = 0
    if total_bytes > settings.max_request_bytes:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.REQUEST_TOO_LARGE.value,
                message=f"Request is {total_bytes} bytes, max allowed is {settings.max_request_bytes}",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 6. schema_version support  6. schema_version 支持
    if request.schema_version not in settings.supported_schema_versions:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.UNSUPPORTED_SCHEMA_VERSION.value,
                message=(
                    f"schema_version '{request.schema_version}' is not supported; "
                    f"supported versions: {sorted(settings.supported_schema_versions)}"
                ),
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 7. Negative time limit  7. 负时间限制
    if request.time_limit_minutes is not None and request.time_limit_minutes < 0:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_TIME_LIMIT.value,
                message=f"time_limit_minutes must be >= 0, got {request.time_limit_minutes}",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # --- P0-05 time semantics ---
    # --- P0-05 时间语义 ---

    # 8. serving_time must be a valid HH:MM string (legacy field).
    # 8. serving_time 必须是合法的 HH:MM 字符串（遗留字段）。
    if request.serving_time is not None:
        parts = request.serving_time.split(":")
        valid = (
            len(parts) == 2
            and parts[0].isdigit()
            and parts[1].isdigit()
            and 0 <= int(parts[0]) <= 23
            and 0 <= int(parts[1]) <= 59
        )
        if not valid:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_SERVING_TIME.value,
                    message=f"serving_time must be HH:MM, got {request.serving_time!r}",
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

    # 9. serving_at must carry a timezone (never treated as local wall-clock).
    # 9. serving_at 必须携带时区（绝不当成本地墙上时钟）。
    if request.serving_at is not None:
        if request.serving_at.tzinfo is None or request.serving_at.utcoffset() is None:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_SERVING_TIME.value,
                    message="serving_at must include a timezone offset",
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

    # 10. Approved decisions (P0-06): structural + revision validation.
    # 10. 已批准决策（P0-06）：结构 + 修订验证。
    if request.approved_decisions:
        from cooking_plan_agent.repair.options import validate_approved_decisions

        issues = validate_approved_decisions(request.approved_decisions, request.plan_revision)
        if issues:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_APPROVED_DECISION.value,
                    message="; ".join(issues),
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

        # Apply decisions as a pure transformation → the downstream pipeline
        # (safety, feasibility, scheduling) re-runs against the RESOLVED
        # request, never bypassing hard rules (P0-06 rule 6).
        # 将决策作为纯变换应用 → 下游管线（安全、可行性、排程）针对已解析的
        # 请求重新运行，绝不绕过硬规则（P0-06 规则 6）。
        from cooking_plan_agent.repair.options import apply_approved_decisions_structured

        resolved = apply_approved_decisions_structured(request, request.approved_decisions)
        if resolved is not request:
            return {"request": resolved}

    return {}


async def parse_recipes_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Extract structured candidates from each recipe's raw text.

    从每份菜谱的原始文本中抽取结构化候选。

    Uses the RecipeExtractor from WorkflowContext (rule-based or LLM-backed).
    Each recipe in the request is individually extracted, then aggregated.

    使用 WorkflowContext 中的 RecipeExtractor（基于规则或基于 LLM）。
    请求中的每份菜谱单独抽取，然后聚合。

    P1-02: multi-recipe extraction runs via ``asyncio.gather`` capped by a
    configurable Semaphore (llm_max_concurrency) and bounded by an overall
    envelope timeout (llm_overall_timeout_seconds) so a single request
    cannot exhaust provider quota or hang the event loop.
    Falls back to rule-based extraction if no extractor is configured.

    P1-02：多菜谱抽取通过 ``asyncio.gather`` 运行，由可配置的 Semaphore
    （llm_max_concurrency）限制并发，并由整体包络超时（llm_overall_timeout_seconds）
    限定，使单个请求无法耗尽 Provider 配额或挂起事件循环。
    若未配置抽取器，则回退到基于规则的抽取。
    """
    import asyncio

    from cooking_plan_agent.config.settings import get_settings

    request = state["request"]

    # Compat layer injects pre-parsed structured candidates (snapshots).
    # Use them directly — never re-invoke the LLM for compat requests
    # (P0-02 rule 4).
    # 兼容层注入已解析的结构化候选（快照）。直接使用它们 —— 绝不为兼容请求
    # 重新调用 LLM（P0-02 规则 4）。
    if request.preparsed_candidates:
        return {"extracted_candidates": request.preparsed_candidates}

    extractor = runtime.context.recipe_extractor
    settings = get_settings()

    candidates: list[ExtractedRecipeCandidate] = []

    if extractor is not None:
        # Use configured extractor (LLM or rule-based from WorkflowContext)
        # 使用已配置的抽取器（LLM 或来自 WorkflowContext 的基于规则抽取器）
        semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        # P1-06 cache is optional — getattr keeps duck-typed contexts working.
        # P1-06 缓存是可选的 —— getattr 使鸭子类型上下文继续可用。
        cache = getattr(runtime.context, "cache", None)

        async def _extract_one(text: str) -> ExtractedRecipeCandidate:
            async with semaphore:
                if cache is None:
                    return await extractor.extract(text)
                # P1-06: reuse stable parse artifacts. Key includes text hash,
                # parser type, model, prompt version, language, schema version
                # so model/prompt upgrades invalidate old entries.
                # P1-06：复用稳定的解析产物。键包含文本哈希、解析器类型、模型、
                # 提示词版本、语言、schema 版本，使模型 / 提示词升级使旧条目失效。
                from typing import cast

                from cooking_plan_agent.infrastructure.cache import build_parse_cache_key
                from cooking_plan_agent.llm.extractor import PARSE_PROMPT_VERSION

                key = build_parse_cache_key(
                    text,
                    parser_type=type(extractor).__name__,
                    model=settings.llm_model,
                    prompt_version=PARSE_PROMPT_VERSION,
                    schema_version=request.schema_version,
                )
                value = await cache.get_or_compute(
                    key,
                    settings.cache_ttl_seconds,
                    lambda: extractor.extract(text),
                )
                return cast(ExtractedRecipeCandidate, value)

        try:
            # Bound the whole batch: a slow provider must not hold the request
            # open beyond llm_overall_timeout_seconds.
            # 限定整批：慢速 Provider 绝不能让请求打开超过 llm_overall_timeout_seconds。
            extracted = await asyncio.wait_for(
                asyncio.gather(*(_extract_one(r.text) for r in request.recipes if r.text)),
                timeout=settings.llm_overall_timeout_seconds,
            )
            candidates = list(extracted)
        except Exception as exc:  # noqa: BLE001 — per-node error → error state
            # 单节点错误 → 错误状态
            # P2-03: public text comes from the catalog; keep only the
            # exception type as controlled diagnostic context.
            # P2-03：公共文本来自目录；仅保留异常类型作为受控诊断上下文。
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.EXTERNAL_PROVIDER_UNAVAILABLE.value,
                    message="Recipe extraction failed",
                    correlation_id=request.request_id,
                    node_name="parse_recipes",
                    diagnostics={"exception_type": type(exc).__name__},
                )
            }
    else:
        # No extractor configured — use built-in rule-based extractor
        # 未配置抽取器 —— 使用内置的基于规则抽取器
        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor

        rule_extractor = RuleExtractor()
        for recipe in request.recipes:
            if recipe.text:
                candidate = await rule_extractor.extract(recipe.text)
                candidates.append(candidate)

    # Async Backend submissions intentionally skip synchronous preprocessing so
    # the client can receive a task handle immediately. In that flow, text gap
    # answers arrive before ``preparsed_candidates`` exist, so validate_input
    # cannot apply them yet. Replay only those field patches against the fresh
    # extraction result; structural decisions were already applied upstream.
    # 异步 Backend 提交有意跳过同步预处理，使客户端能立即收到任务句柄。
    # 在该流程中，文本缺口答复在 ``preparsed_candidates`` 存在之前到达，
    # 因此 validate_input 尚无法应用它们。仅针对最新抽取结果重放这些字段补丁；
    # 结构性决策已在上游应用。
    gap_value_decisions = tuple(
        decision for decision in request.approved_decisions if decision.option_type == "provide_gap_value"
    )
    if candidates and gap_value_decisions:
        from cooking_plan_agent.repair.options import apply_approved_decisions_structured

        extracted_request = request.model_copy(update={"preparsed_candidates": tuple(candidates)})
        resolved_request = apply_approved_decisions_structured(extracted_request, gap_value_decisions)
        candidates = list(resolved_request.preparsed_candidates)

    return {"extracted_candidates": tuple(candidates)}


async def detect_gaps_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Identify missing/inferred fields in extracted candidates.

    识别抽取候选中的缺失 / 推断字段。

    Runs gap detection (find_recipe_gaps) on every extracted candidate.
    Aggregates all gaps across recipes.

    对每个抽取候选运行缺口检测（find_recipe_gaps）。聚合所有菜谱的缺口。
    """
    from cooking_plan_agent.parsing.gaps import find_recipe_gaps

    candidates = state.get("extracted_candidates", ())
    if not candidates:
        return {"gaps": ()}

    all_gaps: list[RecipeGap] = []
    for candidate in candidates:
        gaps = find_recipe_gaps(candidate)
        all_gaps.extend(gaps)

    return {"gaps": tuple(all_gaps)}


async def infer_local_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Apply local cooking rules to fill detected gaps.

    应用本地烹饪规则以填补检测到的缺口。

    Runs infer_local on each candidate with its gaps, then merges inference
    results back into updated candidates. Unresolved critical gaps are left
    in state for routing decisions.

    对每个候选及其缺口运行 infer_local，然后将推断结果合并回更新后的候选。
    未解决的关键缺口保留在状态中供路由决策使用。
    """
    from cooking_plan_agent.parsing.inference import infer_local as local_infer
    from cooking_plan_agent.parsing.inference import merge_inference

    candidates = state.get("extracted_candidates", ())
    gaps = state.get("gaps", ())

    if not candidates or not gaps:
        return {}

    # Partition gaps by recipe_id for per-candidate inference
    # 按 recipe_id 划分缺口，以进行逐候选推断
    gaps_by_recipe: dict[str, list[RecipeGap]] = {}
    for gap in gaps:
        gaps_by_recipe.setdefault(gap.recipe_id, []).append(gap)

    updated_candidates: list[ExtractedRecipeCandidate] = []
    all_unresolved: list[RecipeGap] = []
    all_assumptions: list[Assumption] = []

    for candidate in candidates:
        recipe_gaps = tuple(gaps_by_recipe.get(candidate.recipe_id, ()))
        if not recipe_gaps:
            updated_candidates.append(candidate)
            continue

        result = local_infer(candidate, recipe_gaps)
        updated = merge_inference(candidate, result)
        updated_candidates.append(updated)
        all_unresolved.extend(result.unresolved_gaps)
        all_assumptions.extend(result.assumptions)

    return {
        "extracted_candidates": tuple(updated_candidates),
        "gaps": tuple(all_unresolved),  # Only keep unresolved gaps for routing  仅保留未解决缺口用于路由
    }


async def validate_recipe_ir_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Build validated RecipeIR objects from candidates.

    从候选构建经过验证的 RecipeIR 对象。

    Converts each ExtractedRecipeCandidate → RecipeIR via build_recipe_ir,
    then runs semantic validation (validate_recipe_ir_semantics). Produces
    parsed_recipes in state for downstream scheduling nodes.

    通过 build_recipe_ir 将每个 ExtractedRecipeCandidate 转换为 RecipeIR，
    然后运行语义验证（validate_recipe_ir_semantics）。在状态中产生
    parsed_recipes，供下游排程节点使用。

    If semantic validation fails with errors, returns a workflow error
    that will be routed to FAILED terminal.

    若语义验证因错误失败，返回将被路由到 FAILED 终态的工作流错误。
    """
    from cooking_plan_agent.parsing.ir_builder import (
        attach_research_assumptions,
        build_recipe_ir,
        validate_recipe_ir_semantics,
    )

    candidates = state.get("extracted_candidates", ())
    if not candidates:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                message="No extracted recipe candidates to validate",
                correlation_id=state["request"].request_id,
                node_name="validate_recipe_ir",
            )
        }

    request = state["request"]
    # The extractor produces candidates in the same order as the request's
    # non-empty recipes. Pair each candidate with its request recipe input so
    # the caller's recipe_id overrides the extractor's internal ID and the
    # target_servings drives ingredient scaling (P0-04 rules 2 & 5).
    # 抽取器按请求中非空菜谱的顺序产出候选。将每个候选与其请求菜谱输入配对，
    # 使调用方的 recipe_id 覆盖抽取器的内部 ID，并让 target_servings 驱动
    # 食材缩放（P0-04 规则 2 & 5）。
    request_recipes = tuple(r for r in request.recipes if r.text)
    pairs = tuple(zip(candidates, request_recipes, strict=True))

    # Build RecipeIR from each candidate
    # 从每个候选构建 RecipeIR
    recipes: list[RecipeIR] = []
    for candidate, recipe_input in pairs:
        try:
            recipe_ir = build_recipe_ir(
                candidate,
                request_recipe_id=recipe_input.recipe_id,
                target_servings=recipe_input.target_servings,
            )
            recipes.append(recipe_ir)
        except (ValueError, TypeError) as exc:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                    message="Failed to build RecipeIR",
                    correlation_id=state["request"].request_id,
                    node_name="validate_recipe_ir",
                    diagnostics={"exception_type": type(exc).__name__},
                )
            }

    # P0-06: apply substitute_ingredient decisions as a RecipeIR patch so
    # safety (allergen) and feasibility checks re-run against the NEW
    # ingredient — hard rules are never bypassed (rule 6).
    # P0-06：将 substitute_ingredient 决策作为 RecipeIR 补丁应用，使
    # 安全（过敏原）与可行性检查针对新食材重新运行 —— 硬规则绝不被绕过（规则 6）。
    if request.approved_decisions:
        from cooking_plan_agent.repair.options import apply_ingredient_substitutions_patch

        recipes = list(apply_ingredient_substitutions_patch(tuple(recipes), request.approved_decisions))

    # P1-01: attach evidence-backed research assumptions so provenance is
    # traceable in the final assumption/response.
    # P1-01：附加有证据支撑的研究假设，使来源在最终假设 / 响应中可追溯。
    recipes = list(attach_research_assumptions(tuple(recipes), state.get("research_assumptions", ())))

    # Semantic validation
    # 语义验证
    report = validate_recipe_ir_semantics(tuple(recipes))
    if not report.passed:
        error_details = "; ".join(i.message for i in report.issues if i.severity == "error")
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                message=f"Recipe semantic validation failed: {error_details}",
                correlation_id=state["request"].request_id,
                node_name="validate_recipe_ir",
            )
        }

    return {"parsed_recipes": tuple(recipes)}
