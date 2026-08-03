from pathlib import Path


def test_only_score_once_uses_inference_and_no_prohibited_capabilities_exist() -> None:
    source = Path(__file__).resolve().parents[2] / "src/recommendation_agent"
    workflow = source / "workflow"
    score_references = []
    for path in workflow.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if ".inference.score(" in text:
            score_references.append((path.name, text.count(".inference.score(")))
    assert score_references == [("nodes.py", 1)]
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in workflow.rglob("*.py")).casefold()
    forbidden = (
        "sqlalchemy",
        "sqlite",
        "checkpoint",
        "subprocess",
        "socket",
        "browser",
        "openai",
        "cooking",
        "chatbot",
    )
    assert all(item not in rendered for item in forbidden)
