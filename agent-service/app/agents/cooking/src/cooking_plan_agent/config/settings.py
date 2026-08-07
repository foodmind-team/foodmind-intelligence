# import the required modules
from functools import lru_cache
from typing import Annotated

from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _parse_comma_separated_list(value: object) -> list[str]:
    """Parse a comma-separated env string into a list of stripped items.

    Env sources deliver tuples as a single comma-joined string (e.g.
    COOKING_PLAN_CORS_ALLOW_ORIGINS=http://a,https://b). This validator
    splits on commas; a pre-parsed list passes through unchanged.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


# Comma-separated list setting (used for CORS allow-list, P0-08).
# NoDecode stops pydantic-settings from JSON-decoding the env value first
# (tuple is treated as a complex type); the raw string reaches the validator.
CommaSeparatedList = Annotated[tuple[str, ...], NoDecode, BeforeValidator(_parse_comma_separated_list)]
LOCAL_SERVICE_TOKEN = "local-cooking-token"  # noqa: S105 - rejected outside local


# define the settings model
class Settings(BaseSettings):
    app_name: str = "FoodMind Cooking Plan Agent"
    environment: str = "local"
    log_level: str = "INFO"
    internal_service_token: str = LOCAL_SERVICE_TOKEN
    solver_timeout_seconds: float = 5.0  # the solver timeout in seconds
    max_recipe_count: int = 6  # the maximum number of recipes to return
    max_task_count: int = 100  # the maximum number of tasks to process
    web_research_enabled: bool = False  # whether to enable web research
    # P3-03: solver optimisation depth. "makespan" keeps the legacy Phase-1
    # only solve; "phase12" adds holding minimisation; "full" (default) runs
    # makespan → holding → context-switch. Phase 4 (active labour) stays
    # gated until equivalent execution modes exist in the model.
    solver_optimization_level: str = "full"

    # --- Regional food-safety policy (P3-04) ---
    # Deployment-level default region when the request does not explicitly
    # select one. An unknown region (here or in the request) is a hard error —
    # the service never silently falls back (D6). Supported regions are
    # defined by the registered packs in safety/policies/.
    safety_policy_region: str = "US"
    # Optional explicit policy version. None resolves the latest registered
    # version of the region; older versions remain queryable for audit but are
    # not selected by default.
    safety_policy_version: str | None = None

    # --- Internal API security (P0-08) ---
    # CORS is DISABLED by default for internal APIs. If a caller genuinely
    # needs browser cross-origin access, list explicit origins here —
    # wildcards are rejected at startup (credentials + "*" is prohibited).
    cors_allow_origins: CommaSeparatedList = ()
    # Minimum length required for the internal service token in non-local
    # environments (P0-08 rule 4). local/CI may use short test tokens.
    min_service_token_length: int = 16

    # --- Request validation limits (P0-03) ---
    # Hard caps applied at the workflow input boundary. All values are
    # configurable via environment variables with the COOKING_PLAN_ prefix.
    # Byte limits are measured on UTF-8 encoded payloads.
    max_recipe_text_bytes: int = 50_000  # per-recipe raw text byte cap
    max_request_bytes: int = 1_000_000  # total serialised request byte cap
    supported_schema_versions: tuple[str, ...] = ("1.0",)  # accepted schema_version values

    # --- Request-level backpressure (P1-03) ---
    # Two-layer limiter: at most `max_active_requests` requests run
    # concurrently; excess requests queue for `queue_timeout_seconds`. When
    # the queue is full or a waiter times out, the request is rejected with
    # 503 + Retry-After instead of unboundedly piling up. Health endpoints
    # bypass this limiter so orchestrators can always probe the process.
    max_active_requests: int = 20
    max_queued_requests: int = 100
    queue_timeout_seconds: float = 5.0

    # --- LLM integration (local Ollama via OpenAI-compatible API) ---
    # Provider-neutral: base_url + model are configurable so any OpenAI-
    # compatible endpoint (Ollama, localhost proxy, cloud) can be swapped in.
    llm_enabled: bool = True  # master switch; False keeps rule-based pipeline
    llm_base_url: str = "https://api.deepseek.com"  # OpenAI-compatible base URL
    llm_model: str = "deepseek-chat"  # model name
    llm_api_key: str | None = None  # bearer token for cloud providers (Ollama: None)
    # Recipe extraction returns a sizeable JSON document.  Allow cloud models
    # enough time to complete it; callers with interactive deadlines can
    # override this through COOKING_PLAN_LLM_TIMEOUT_SECONDS.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2  # retries before falling back to rule-based
    llm_temperature: float = 0.1  # low temp for deterministic structured output
    # Bounds provider output so a malformed/overly verbose structured response
    # cannot consume the entire request timeout.
    llm_max_output_tokens: int = 2048

    # --- LLM concurrency & connection budget (P1-02) ---
    # Max in-flight LLM calls per request (recipe extraction fan-out). The
    # multi-recipe gather is capped by this semaphore so a single request
    # cannot exhaust the provider quota.
    llm_max_concurrency: int = 4
    # httpx connection pool size: total connections the lifecycle-level
    # AsyncClient may hold. One shared client is reused across all requests
    # instead of creating a fresh pool per call.
    llm_connection_pool_size: int = 10
    # Overall envelope timeout for the whole multi-recipe extraction batch.
    # Per-call timeout is llm_timeout_seconds; this bounds the gather.
    llm_overall_timeout_seconds: float = 240.0

    # --- Schedule explanation (P2-02 / P4-01) ---
    # READY responses may carry a short "why this schedule" explanation
    # produced by the LLM explainer, with a deterministic fallback. Disabled
    # by default so CI stays offline-deterministic; this is also the
    # feature-flag rollback for the capability. The explanation is additive
    # and never alters the verified schedule.
    explanation_enabled: bool = False

    # P5-3: schedule repair loop（反思修复循环）。
    # verify 失败后最多重试次数；0 表示禁用（保持原 FAILED 语义）。
    schedule_repair_max_attempts: int = 2
    # 重试前是否允许 LLM 做诊断摘要（仅建议，最终动作仍由规则裁决）。
    schedule_repair_llm_enabled: bool = False

    # P5-2: ReAct 控制器循环上限；0 表示不启用控制器（保持原 DAG）。
    agent_max_steps: int = 5
    agent_controller_enabled: bool = False

    # P5-4: 多轮确认对话。NEEDS_CONFIRMATION 由终态升级为中间态：
    # 用户 answers 后经 apply_confirmation 续接重排。默认关闭保持原终态。
    confirmation_dialog_enabled: bool = False

    # --- Bounded web research controls (handbook 10.1, 10.9) ---
    # Per-query timeout in seconds — search fails to confirmation on timeout
    research_timeout_seconds: float = 10.0
    # At most N queries per dish in MVP (handbook 10.9)
    research_max_queries_per_dish: int = 2
    # At most N results per query
    research_max_results_per_query: int = 3
    # Domain allow-list for web research (handbook 10.4).
    # Separate source classes are maintained in research/config.py,
    # but the raw list lives here for environment-variable configuration.
    allowed_research_domains: list[str] = []
    # Median Absolute Deviation threshold for duration reconciliation
    # (handbook 10.7). If MAD exceeds this fraction of the median,
    # the result is flagged as needs_confirmation.
    research_disagreement_threshold: float = 0.5

    # --- Intermediate artifact cache (P1-06) ---
    # Caches stable parse/research artifacts (never final READY responses).
    # Disabling only affects performance, never results. In-memory and
    # instance-level; distributed cache is a P3-02 concern.
    cache_enabled: bool = False
    cache_ttl_seconds: float = 3600.0
    cache_max_entries: int = 1000
    # Oversized artifacts are skipped rather than cached (memory bound).
    cache_max_item_bytes: int = 100_000

    # --- Shared preparation merging (P2-01) ---
    # Merges identical ingredient-prep operations across recipes via the
    # preparation trie (one wash instead of N). Disabling restores per-recipe
    # preparation and is the rollback path for the feature.
    shared_prep_enabled: bool = True

    # --- Workflow checkpoint persistence (P2-06) ---
    # Persists PlanState at node boundaries via a LangGraph checkpointer so
    # long tasks, human confirmation, and async execution (P3-01) can resume
    # after a process restart. Disabled by default — the existing stateless
    # execution mode is preserved exactly.
    checkpoint_enabled: bool = False
    # Backend selection: "memory" (InMemorySaver, tests/CI) or "sqlite"
    # (AsyncSqliteSaver, local dev + MVP). Postgres saver is a P3-02 concern.
    checkpoint_backend: str = "sqlite"
    # SQLite file for AsyncSqliteSaver when checkpoint_backend="sqlite".
    checkpoint_sqlite_path: str = "data/checkpoints.sqlite"
    # Retention window for checkpoint rows (operational cleanup / audit).
    checkpoint_ttl_days: int = 30

    # --- Async task API (P3-01) ---
    # Long-running generation is submitted as a task and executed by the
    # in-process worker; the synchronous v1/native endpoints are preserved.
    task_api_enabled: bool = False
    # SQLite file backing the task repository (P3-01 MVP storage).
    task_db_path: str = "data/tasks.sqlite"
    # Default task TTL; tasks exceeding it move to EXPIRED.
    task_default_ttl_seconds: int = 3600
    # Max tasks the in-process worker executes concurrently.
    task_worker_concurrency: int = 2

    # --- Async task SSE progress (P4-04) ---
    # Optional Server-Sent-Events progress stream per task. Enabled by
    # default when the task API is on; the polling endpoints stay available
    # regardless and remain the fallback. When disabled, GET /tasks/{id}/events
    # returns 404 and polling is unaffected.
    task_sse_enabled: bool = True

    # --- Distributed task queue (P4-05) ---
    # Worker queue backend. Stage A ships the in-process queue (approved MVP
    # path, no behaviour change). Distributed backends (e.g. "redis") are
    # Stage B and must NOT be enabled until the queue infrastructure
    # (ADR + approval) lands — the service refuses to start with an
    # unapproved backend instead of silently degrading.
    task_queue_backend: str = "inprocess"

    # define the model config
    model_config = SettingsConfigDict(
        env_prefix="COOKING_PLAN_",
        env_file=".env",
        extra="forbid",  # if the env file contains extra variables, throw an error
    )


@lru_cache
def get_settings() -> Settings:
    # model_validate({}) keeps mypy happy (internal_service_token has no
    # default) while still reading all COOKING_PLAN_* env sources.
    return Settings.model_validate({})
