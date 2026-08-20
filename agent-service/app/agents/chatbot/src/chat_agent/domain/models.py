"""Pydantic mirrors of the Spring Boot chat-agent-v2 wire contract."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=False,
        extra="forbid",
        str_strip_whitespace=True,
    )


BoundedText = Annotated[str, StringConstraints(min_length=1, max_length=12000)]
TurnText = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
SuggestionText = Annotated[str, StringConstraints(min_length=1, max_length=200)]
SourceType = Literal["FOOD_RECORD", "FOOD_PRODUCT", "PLACE"]
ResponseStatus = Literal["SUCCEEDED", "FALLBACK_SUCCEEDED", "UNSUPPORTED"]
Destination = Literal[
    "INVENTORY",
    "SHOPPING_LISTS",
    "SAVED_RECIPES",
    "COOKING_PLANS",
    "RECOMMENDATIONS",
    "EXPLORE",
]


class SharedReference(WireModel):
    reference_id: UUID
    source_type: SourceType
    source_id: UUID
    title: Annotated[str | None, StringConstraints(max_length=500)] = None
    snippet: Annotated[str | None, StringConstraints(max_length=4000)] = None


class ConversationTurn(WireModel):
    role: Literal["USER", "ASSISTANT"]
    content: TurnText


class AgentChatRequest(WireModel):
    contract_version: Literal["chat-agent-v2"]
    request_id: UUID
    session_id: UUID
    user_message_id: UUID
    trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    expires_at: datetime | None = None
    message: BoundedText
    delegation_token: Annotated[str | None, StringConstraints(max_length=8192)] = Field(default=None, repr=False)
    shared_references: Annotated[list[SharedReference], Field(max_length=20)] = Field(default_factory=list)
    recent_turns: Annotated[list[ConversationTurn], Field(max_length=8)] = Field(default_factory=list)


class ChatSource(WireModel):
    source_type: SourceType
    source_id: UUID
    sequence_no: Annotated[int, Field(ge=1, le=10)]
    grounding_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentChatResponse(WireModel):
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    contract_version: Literal["chat-agent-v2"] = "chat-agent-v2"
    request_id: UUID
    session_id: UUID
    user_message_id: UUID
    trace_id: str
    agent_trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    response_status: ResponseStatus
    answer: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    sources: Annotated[list[ChatSource], Field(max_length=10)] = Field(default_factory=list)
    suggested_questions: Annotated[list[SuggestionText], Field(max_length=3)] = Field(default_factory=list)
    suggested_destinations: Annotated[list[Destination], Field(max_length=3)] = Field(default_factory=list)


class ErrorEnvelope(BaseModel):
    status: int
    error_code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class GroundedSource:
    source_type: SourceType
    source_id: UUID
    title: str | None = None
    subtitle: str | None = None
    snippet: str | None = None
    occurred_at: str | None = None
    grounding_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_reference(cls, reference: SharedReference) -> "GroundedSource":
        return cls(
            source_type=reference.source_type,
            source_id=reference.source_id,
            title=reference.title,
            snippet=reference.snippet,
            grounding_metadata={"referenceId": str(reference.reference_id), "origin": "shared_reference"},
        )
