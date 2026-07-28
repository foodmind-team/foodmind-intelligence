#import the required modules
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


#define the settings model
class Settings(BaseSettings):
    app_name:str = "FoodMind Cooking Plan Agent"
    environment:str = "local"
    log_level:str = "INFO"
    internal_service_token:str
    solver_timeout_seconds:float = 5.0#the solver timeout in seconds
    max_recipe_count:int = 6#the maximum number of recipes to return
    max_task_count:int = 100#the maximum number of tasks to process
    web_research_enabled:bool = False#whether to enable web research
    #define the model config
    model_config = SettingsConfigDict(
        env_prefix="COOKING_PLAN_",
        env_file=".env",
        extra="forbid",#if the env file contains extra variables, throw an error
    )

@lru_cache
def get_settings():
    return Settings()
