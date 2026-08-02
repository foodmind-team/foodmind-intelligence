# import the required modules
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# define the settings model
class Settings(BaseSettings):
    app_name: str = "FoodMind Cooking Plan Agent"
    environment: str = "local"
    log_level: str = "INFO"
    internal_service_token: str
    solver_timeout_seconds: float = 5.0  # the solver timeout in seconds
    max_recipe_count: int = 6  # the maximum number of recipes to return
    max_task_count: int = 100  # the maximum number of tasks to process
    web_research_enabled: bool = False  # whether to enable web research

    # --- Request validation limits (P0-03) ---
    # Hard caps applied at the workflow input boundary. All values are
    # configurable via environment variables with the COOKING_PLAN_ prefix.
    # Byte limits are measured on UTF-8 encoded payloads.
    max_recipe_text_bytes: int = 50_000  # per-recipe raw text byte cap
    max_request_bytes: int = 1_000_000  # total serialised request byte cap
    supported_schema_versions: tuple[str, ...] = ("1.0",)  # accepted schema_version values

    # --- LLM integration (local Ollama via OpenAI-compatible API) ---
    # Provider-neutral: base_url + model are configurable so any OpenAI-
    # compatible endpoint (Ollama, localhost proxy, cloud) can be swapped in.
    llm_enabled: bool = False  # master switch; False keeps rule-based pipeline
    llm_base_url: str = "https://api.deepseek.com"  # OpenAI-compatible base URL
    llm_model: str = "deepseek-chat"  # model name
    llm_api_key: str | None = None  # bearer token for cloud providers (Ollama: None)
    llm_timeout_seconds: float = 30.0  # per-call timeout
    llm_max_retries: int = 2  # retries before falling back to rule-based
    llm_temperature: float = 0.1  # low temp for deterministic structured output

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
