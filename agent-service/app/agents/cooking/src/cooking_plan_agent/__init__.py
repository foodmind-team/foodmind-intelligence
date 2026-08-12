def main() -> None:
    import uvicorn

    uvicorn.run("cooking_plan_agent.main:app", host="0.0.0.0", port=8003, log_config=None)  # noqa: S104
