from conftest import AGENT_FIXTURES, load_json
from fastapi.testclient import TestClient


def test_missing_wrong_scheme_and_bad_credential_are_safe(client: TestClient) -> None:
    payload = load_json(AGENT_FIXTURES / "valid-normal.json")
    cases: tuple[tuple[dict[str, str], str], ...] = (
        ({}, "MISSING_AUTHORIZATION_HEADER"),
        ({"Authorization": "Basic abc"}, "INVALID_AUTHORIZATION_SCHEME"),
        ({"Authorization": "Bearer wrong-secret-canary"}, "INVALID_INTERNAL_CREDENTIAL"),
    )
    for headers, expected_code in cases:
        response = client.post("/internal/v1/recommendations/generate", json=payload, headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == expected_code
        assert "wrong-secret-canary" not in response.text


def test_valid_auth_is_canonical_not_ready(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/internal/v1/recommendations/generate",
        json=load_json(AGENT_FIXTURES / "valid-normal.json"),
        headers=auth_headers,
    )
    assert response.status_code == 503
    assert response.json()["error"] == {"code": "SERVICE_NOT_READY", "retryable": True}
