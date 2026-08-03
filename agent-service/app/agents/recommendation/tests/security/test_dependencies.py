import tomllib
from pathlib import Path


def test_runtime_dependency_surface_is_allow_listed() -> None:
    project = Path(__file__).resolve().parents[2]
    configuration = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        item.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0] for item in configuration["project"]["dependencies"]
    }
    assert dependencies == {"fastapi", "httpx", "langgraph", "pydantic", "pydantic-settings", "uvicorn"}
    forbidden = {"sqlalchemy", "psycopg", "pymysql", "pandas", "sklearn", "torch", "openai", "boto3"}
    assert not dependencies.intersection(forbidden)
