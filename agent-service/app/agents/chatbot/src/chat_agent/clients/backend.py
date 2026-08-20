"""Bounded client for Backend-owned, delegation-scoped chat tools."""

from __future__ import annotations

import json
import re
from typing import Any, cast
from uuid import UUID

import httpx

from chat_agent.config.settings import Settings
from chat_agent.domain.models import GroundedSource, SourceType

_SOURCE_TYPES = {"FOOD_RECORD", "FOOD_PRODUCT", "PLACE"}
_PROFILE_SCALAR_FIELDS = {
    "budgetMin",
    "budgetMax",
    "currency",
    "spiceTolerance",
    "preferredArea",
    "maxDistanceKm",
    "foodGoal",
    "drinkSweetnessPreference",
    "drinkIcePreference",
    "cookingRegion",
}
_PROFILE_CODE_LIST_FIELDS = {
    "likedCuisineCodes",
    "dislikedCuisineCodes",
    "dietaryTagCodes",
    "preferredMealTypes",
}


class BackendToolError(Exception):
    """Safe failure at the Backend tool boundary."""


class BackendToolClient:
    def __init__(self, *, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def search(
        self,
        *,
        query: str,
        delegation_token: str,
        timeout_seconds: float,
    ) -> tuple[GroundedSource, ...]:
        raw = await self._post(
            "/internal/v1/search",
            {"query": query[:200], "sourceTypes": [], "after": None, "size": 10},
            delegation_token=delegation_token,
            timeout_seconds=timeout_seconds,
        )
        sources, has_next = _search_result(raw)
        # A collection question (for example, "how many restaurants") has no
        # concrete text to match against every catalogue title. It remains a
        # delegated read-only request, so broaden only the bounded collection.
        if not sources and _is_collection_query(query):
            return await self.explore(
                source_types=["PLACE"] if _is_place_collection(query) else [],
                delegation_token=delegation_token,
                timeout_seconds=timeout_seconds,
            )
        return tuple(_with_page_metadata(source, has_next) for source in sources)

    async def explore(
        self,
        *,
        source_types: list[SourceType],
        delegation_token: str,
        timeout_seconds: float,
    ) -> tuple[GroundedSource, ...]:
        raw = await self._post(
            "/internal/v1/explore",
            {"sourceTypes": source_types, "after": None, "size": 10},
            delegation_token=delegation_token,
            timeout_seconds=timeout_seconds,
        )
        sources, has_next = _search_result(raw)
        return tuple(_with_page_metadata(source, has_next) for source in sources)

    async def resolve(
        self,
        *,
        session_id: UUID,
        reference_ids: list[UUID],
        delegation_token: str,
        timeout_seconds: float,
    ) -> tuple[GroundedSource, ...]:
        raw = await self._post(
            "/internal/v1/references/resolve",
            {"sessionId": str(session_id), "referenceIds": [str(item) for item in reference_ids]},
            delegation_token=delegation_token,
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(raw, list):
            raise BackendToolError("Backend reference response is invalid")
        return tuple(_resolved_source(item) for item in raw if isinstance(item, dict) and item.get("available") is True)

    async def profile(
        self,
        *,
        delegation_token: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        raw = await self._get(
            "/internal/v1/profile",
            delegation_token=delegation_token,
            timeout_seconds=timeout_seconds,
        )
        return _profile_response(raw)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return the small, read-only registry exposed to the provider."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the current user's authorised FoodMind records, products, and places. "
                    "Build a self-contained query from the conversation.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explore",
                    "description": "List the current user's authorised FoodMind items when a bounded collection is "
                    "needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sourceTypes": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["FOOD_RECORD", "FOOD_PRODUCT", "PLACE"]},
                                "maxItems": 3,
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "resolve",
                    "description": "Resolve references that the user has shared in this FoodMind chat session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "referenceIds": {
                                "type": "array",
                                "items": {"type": "string", "format": "uuid"},
                                "minItems": 1,
                                "maxItems": 20,
                            }
                        },
                        "required": ["referenceIds"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def execute_tool_call(
        self,
        *,
        name: str,
        arguments: str,
        session_id: UUID,
        delegation_token: str,
        timeout_seconds: float,
    ) -> tuple[GroundedSource, ...]:
        """Validate and dispatch a provider-requested tool without widening authority."""
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise BackendToolError("Tool arguments are invalid JSON") from exc
        if not isinstance(payload, dict):
            raise BackendToolError("Tool arguments must be an object")
        if name == "search":
            query = payload.get("query")
            if set(payload) != {"query"} or not isinstance(query, str) or not query.strip():
                raise BackendToolError("Search arguments are invalid")
            return await self.search(query=query, delegation_token=delegation_token, timeout_seconds=timeout_seconds)
        if name == "explore":
            source_types = payload.get("sourceTypes", [])
            if set(payload) != {"sourceTypes"} or not isinstance(source_types, list) or len(source_types) > 3:
                raise BackendToolError("Explore arguments are invalid")
            if not all(isinstance(item, str) and item in _SOURCE_TYPES for item in source_types):
                raise BackendToolError("Explore arguments are invalid")
            return await self.explore(
                source_types=cast(list[SourceType], source_types),
                delegation_token=delegation_token,
                timeout_seconds=timeout_seconds,
            )
        if name == "resolve":
            reference_ids = payload.get("referenceIds")
            if (
                set(payload) != {"referenceIds"}
                or not isinstance(reference_ids, list)
                or not 1 <= len(reference_ids) <= 20
            ):
                raise BackendToolError("Resolve arguments are invalid")
            try:
                parsed_ids = [UUID(str(item)) for item in reference_ids]
            except (TypeError, ValueError) as exc:
                raise BackendToolError("Resolve arguments are invalid") from exc
            return await self.resolve(
                session_id=session_id,
                reference_ids=parsed_ids,
                delegation_token=delegation_token,
                timeout_seconds=timeout_seconds,
            )
        raise BackendToolError("Tool is not registered")

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        delegation_token: str,
        timeout_seconds: float,
    ) -> Any:
        try:
            response = await self._client.post(
                path,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._settings.backend_service_token.get_secret_value()}",
                    "X-FoodMind-Delegation": f"Bearer {delegation_token}",
                    "Content-Type": "application/json",
                },
                timeout=min(timeout_seconds, self._settings.backend_timeout_seconds),
            )
            response.raise_for_status()
            if len(response.content) > self._settings.backend_max_response_bytes:
                raise BackendToolError("Backend tool response is too large")
            return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise BackendToolError("Backend tool request failed") from exc

    async def _get(
        self,
        path: str,
        *,
        delegation_token: str,
        timeout_seconds: float,
    ) -> Any:
        try:
            response = await self._client.get(
                path,
                headers={
                    "Authorization": f"Bearer {self._settings.backend_service_token.get_secret_value()}",
                    "X-FoodMind-Delegation": f"Bearer {delegation_token}",
                },
                timeout=min(timeout_seconds, self._settings.backend_timeout_seconds),
            )
            response.raise_for_status()
            if len(response.content) > self._settings.backend_max_response_bytes:
                raise BackendToolError("Backend tool response is too large")
            return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise BackendToolError("Backend tool request failed") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _source_type(value: Any) -> SourceType:
    if not isinstance(value, str) or value not in _SOURCE_TYPES:
        raise BackendToolError("Backend returned an unsupported source type")
    return cast(SourceType, value)


def _search_source(value: Any) -> GroundedSource:
    if not isinstance(value, dict):
        raise BackendToolError("Backend search item is invalid")
    try:
        return GroundedSource(
            source_type=_source_type(value.get("sourceType")),
            source_id=UUID(str(value.get("sourceId"))),
            title=value.get("title") if isinstance(value.get("title"), str) else None,
            subtitle=value.get("subtitle") if isinstance(value.get("subtitle"), str) else None,
            snippet=value.get("snippet") if isinstance(value.get("snippet"), str) else None,
            occurred_at=value.get("occurredAt") if isinstance(value.get("occurredAt"), str) else None,
            grounding_metadata={"origin": "backend_search"},
        )
    except (TypeError, ValueError) as exc:
        raise BackendToolError("Backend search item is invalid") from exc


def _search_result(raw: Any) -> tuple[tuple[GroundedSource, ...], bool]:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise BackendToolError("Backend search response is invalid")
    return (
        tuple(_search_source(item) for item in cast(list[Any], raw["items"])),
        raw.get("hasNext") is True,
    )


def _profile_response(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise BackendToolError("Backend profile response is invalid")
    profile: dict[str, Any] = {}
    for field in _PROFILE_SCALAR_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if field in {"budgetMin", "budgetMax", "maxDistanceKm"}:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BackendToolError("Backend profile response is invalid")
        elif field == "spiceTolerance":
            if not isinstance(value, int) or isinstance(value, bool):
                raise BackendToolError("Backend profile response is invalid")
        elif not isinstance(value, str):
            raise BackendToolError("Backend profile response is invalid")
        profile[field] = value
    for field in _PROFILE_CODE_LIST_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise BackendToolError("Backend profile response is invalid")
        profile[field] = value
    allergens = raw.get("allergens")
    if allergens is not None:
        if not isinstance(allergens, list):
            raise BackendToolError("Backend profile response is invalid")
        parsed_allergens: list[dict[str, str]] = []
        for allergen in allergens:
            if not isinstance(allergen, dict):
                raise BackendToolError("Backend profile response is invalid")
            code, severity = allergen.get("code"), allergen.get("severity")
            if not isinstance(code, str) or not isinstance(severity, str):
                raise BackendToolError("Backend profile response is invalid")
            parsed_allergens.append({"code": code, "severity": severity})
        profile["allergens"] = parsed_allergens
    return profile


def _with_page_metadata(source: GroundedSource, has_next: bool) -> GroundedSource:
    return GroundedSource(
        source_type=source.source_type,
        source_id=source.source_id,
        title=source.title,
        subtitle=source.subtitle,
        snippet=source.snippet,
        occurred_at=source.occurred_at,
        grounding_metadata={**source.grounding_metadata, "hasNext": has_next},
    )


def _is_collection_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:restaurant|restaurants|place|places|catalogue|catalog)\b|餐厅|饭店|地点|场所",
            query,
            re.IGNORECASE,
        )
    )


def _is_place_collection(query: str) -> bool:
    return bool(re.search(r"\b(?:restaurant|restaurants|place|places)\b|餐厅|饭店|地点|场所", query, re.IGNORECASE))


def _resolved_source(value: dict[str, Any]) -> GroundedSource:
    try:
        reference_id = UUID(str(value.get("referenceId")))
        return GroundedSource(
            source_type=_source_type(value.get("sourceType")),
            source_id=UUID(str(value.get("sourceId"))),
            title=value.get("title") if isinstance(value.get("title"), str) else None,
            snippet=value.get("snippet") if isinstance(value.get("snippet"), str) else None,
            grounding_metadata={"referenceId": str(reference_id), "origin": "backend_reference_resolve"},
        )
    except (TypeError, ValueError) as exc:
        raise BackendToolError("Backend reference item is invalid") from exc
