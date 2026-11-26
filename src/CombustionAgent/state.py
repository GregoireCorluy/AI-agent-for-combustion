"""Serializable session state shared with the orchestrator and tools."""
from typing import Any, Literal

from pydantic import BaseModel, Field

from .parameters import InputParameters


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ToolEvent(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Authoritative current task state; histories are supporting context."""

    project_id: str | None = None
    user_message: str = ""
    input_parameters: InputParameters = Field(default_factory=InputParameters)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    tool_history: list[ToolEvent] = Field(default_factory=list)
    validation_messages: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    reduction_results: list[dict[str, Any]] = Field(default_factory=list)
    recommended_mechanism: str | None = None
    status: Literal["collecting_inputs", "ready", "running", "completed", "error"] = "collecting_inputs"
    pending_question: str | None = None

    def orchestrator_view(self) -> dict[str, Any]:
        """Return a compact, JSON-safe view for the orchestrator prompt."""
        return {
            "status": self.status,
            "latest_user_message": self.user_message,
            "input_parameters": self.input_parameters.model_dump(mode="json"),
            "missing_required_fields": self.missing_required_fields,
            "validation_messages": self.validation_messages[-12:],
            "validation_errors": self.validation_errors[-12:],
            "reduction_results": self.reduction_results,
            "recommended_mechanism": self.recommended_mechanism,
            "recent_conversation": [m.model_dump() for m in self.conversation_history[-8:]],
            "recent_tool_results": [e.model_dump(mode="json") for e in self.tool_history[-5:]],
        }
