from recommendation_agent.observability.redaction import redact


def test_recursive_redaction_fails_closed_for_nested_sensitive_names() -> None:
    canary = "nested-sensitive-canary"
    value = redact(
        {
            "safe": "visible",
            "items": [{"offeringKey": canary}, {"requestBody": canary}],
            "exception": {"message": canary},
        }
    )
    rendered = repr(value)
    assert canary not in rendered
    assert "visible" in rendered
