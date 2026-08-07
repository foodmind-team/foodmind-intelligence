from datetime import UTC, datetime, timedelta

from fixtures.fake_inference import fixture_agent_client


def test_v1_local_compatibility_route_uses_v2_workflow_and_inference() -> None:
    with fixture_agent_client(enable_v1_compatibility=True) as (client, fake):
        response = client.post(
            "/internal/compat/v1/recommendations/generate",
            headers={"Authorization": "Bearer e2e-agent-token"},
            json={
                "contractVersion": "recommendation-agent-v1",
                "requestId": "30000000-0000-4000-8000-000000000001",
                "sessionId": "30000000-0000-4000-8000-000000000002",
                "traceId": "30000000-0000-4000-8000-000000000003",
                "deadlineAt": (datetime.now(UTC) + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
                "requestContext": {"mealType": "DINNER"},
                "preferenceContext": {"likedCuisineCodes": ["CHINESE"]},
                "candidates": [
                    {
                        "candidateId": f"30000000-0000-4000-8000-00000000010{index}",
                        "placeMealId": f"40000000-0000-4000-8000-00000000010{index}",
                        "features": {
                            "mealType": "DINNER",
                            "cuisineCode": "CHINESE" if index == 1 else "MALAY",
                            "wantToTry": index == 2,
                            "personalRecordCount": index - 1,
                            "groupRecordCount": 3 if index == 3 else 0,
                            "groupAverageRating": 4.5 if index == 3 else None,
                            "distanceKm": 1.0 + index,
                        },
                    }
                    for index in range(1, 4)
                ],
            },
        )

    assert response.status_code == 200
    assert fake.attempts == 1
    body = response.json()
    assert body["contractVersion"] == "recommendation-agent-v1"
    assert body["status"] == "SUCCEEDED"
    assert body["featureSchemaVersion"] == "recommendation-features-v1"
    assert 1 <= len(body["candidates"]) <= 3
    assert [candidate["rank"] for candidate in body["candidates"]] == list(range(1, len(body["candidates"]) + 1))


def test_v1_compatibility_route_is_disabled_by_default() -> None:
    with fixture_agent_client() as (client, fake):
        response = client.post(
            "/internal/compat/v1/recommendations/generate",
            headers={"Authorization": "Bearer e2e-agent-token"},
            json={},
        )
    assert response.status_code == 404
    assert fake.attempts == 0


def test_canonical_route_accepts_backend_v1_envelope_during_migration() -> None:
    with fixture_agent_client() as (client, fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            headers={"Authorization": "Bearer e2e-agent-token"},
            json={
                "contractVersion": "recommendation-agent-v1",
                "requestId": "30000000-0000-4000-8000-000000000001",
                "sessionId": "30000000-0000-4000-8000-000000000002",
                "traceId": "30000000-0000-4000-8000-000000000003",
                "deadlineAt": (datetime.now(UTC) + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
                "requestContext": {"mealType": "DINNER"},
                "preferenceContext": {"likedCuisineCodes": ["CHINESE"]},
                "candidates": [
                    {
                        "candidateId": f"30000000-0000-4000-8000-00000000010{index}",
                        "placeMealId": f"40000000-0000-4000-8000-00000000010{index}",
                        "features": {
                            "mealType": "DINNER",
                            "cuisineCode": "CHINESE" if index == 1 else "MALAY",
                            "wantToTry": index == 2,
                            "personalRecordCount": index - 1,
                            "groupRecordCount": 3 if index == 3 else 0,
                            "groupAverageRating": 4.5 if index == 3 else None,
                            "distanceKm": 1.0 + index,
                        },
                    }
                    for index in range(1, 4)
                ],
            },
        )

    assert response.status_code == 200
    assert fake.attempts == 1
    assert response.json()["contractVersion"] == "recommendation-agent-v1"
    assert response.json()["status"] == "SUCCEEDED"
