from recommendation_agent.observability.redaction import redact


def test_deep_redaction_covers_keys_bodies_and_uris() -> None:
    canary = "CANARY-DO-NOT-LOG"
    redacted = redact(
        {
            "authorization": f"Bearer {canary}",
            "nested": {"modelUserKey": canary, "featureVector": [canary]},
            "message": f"call https://example.test/path?token={canary}",
        }
    )
    rendered = repr(redacted)
    assert canary not in rendered
    assert "[REDACTED]" in rendered
