from pathlib import Path


def test_recommendation_agent_has_no_forbidden_runtime_imports() -> None:
    source = Path(__file__).resolve().parents[2] / "src/recommendation_agent"
    forbidden = (
        "cooking_plan_agent",
        "chatbot",
        "sqlalchemy",
        "langchain",
        "sklearn",
        "pandas",
        "subprocess",
    )
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in source.rglob("*.py")).casefold()
    assert all(item not in rendered for item in forbidden)
