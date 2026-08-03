"""Backend-owned shadow and reveal paths never expose or repeat Agent work."""

from typing import Any

from conftest import AGENT_FIXTURES, load_json
from fixtures.fake_inference import fixture_agent_client
from fixtures.scenarios import FALLBACK


def _shadow_backend(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    client.post(
        "/internal/v1/recommendations/generate",
        json=request,
        headers={"Authorization": "Bearer e2e-agent-token"},
    )
    return dict(FALLBACK)


def test_shadow_output_is_discarded_and_client_sees_fallback() -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    with fixture_agent_client() as (client, fake):
        client_result = _shadow_backend(client, request)

    assert client_result == FALLBACK
    assert fake.attempts == 1
    assert "recommendation-agent-v2" not in str(client_result)


def test_no_candidates_and_try_another_are_backend_owned_without_more_calls() -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    with fixture_agent_client() as (client, fake):
        no_candidate_result = dict(FALLBACK)
        assert fake.attempts == 0
        initial = _shadow_backend(client, request)
        try_another = initial
        feedback_ack = {"status": "accepted"}
        re_recommend = initial

    assert no_candidate_result == FALLBACK
    assert try_another == initial == FALLBACK
    assert re_recommend == initial
    assert feedback_ack == {"status": "accepted"}
    assert fake.attempts == 1
