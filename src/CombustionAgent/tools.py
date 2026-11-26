"""Extensible tool registry and initial CombustionAgent tools."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field
from jsonschema import validate as validate_json_schema

from .agent_tool import AgentToolMechReduction
from .model_manager import ModelManager
from .parameter_validation import ParameterValidator
from .parameters import InputParameters
from .state import AgentState


class ToolCall(BaseModel):
    action: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    response: str | None = None


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], AgentState], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool {name!r}") from exc

    def descriptions(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description,
                 "parameters": t.parameters} for t in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any], state: AgentState) -> dict[str, Any]:
        tool = self.get(name)
        validate_json_schema(instance=arguments, schema=tool.parameters)
        return tool.handler(arguments, state)


def _json_from_reply(reply: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating a fenced response."""
    cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a valid JSON object")


def _json_safe(value: Any) -> Any:
    """Convert common NumPy and Path values from scientific metrics to JSON types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


class CombustionTools:
    """Domain tool implementations, independent of the orchestration loop."""

    REQUIRED_FOR_DRGEP = (
        "mechanism", "fuel", "pressure_start", "pressure_end", "pressure_unit",
        "temperature_autoignition_start", "temperature_autoignition_end",
        "temperature_unit", "equivalence_ratio_start", "equivalence_ratio_end",
        "target_species",
    )

    def __init__(self, models: ModelManager, database_path: str):
        self.models = models
        self.database_path = database_path
        self.validator = ParameterValidator(database_path)

    def register(self, registry: ToolRegistry) -> None:
        registry.register(RegisteredTool(
            name="retrieve_input_parameters",
            description=("Extract combustion parameters from the user's latest message with the "
                         "specialized extractor, merge explicitly supplied fields into current state, "
                         "then run deterministic mechanism, unit, regime, fuel and species checks."),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=self.retrieve_input_parameters,
        ))
        registry.register(RegisteredTool(
            name="modify_input_parameters",
            description=("Use the parameter editing model to update the existing InputParameters "
                         "from a user correction or a tool diagnostic. Run deterministic checks afterward."),
            parameters={"type": "object", "properties": {
                "instruction": {"type": "string", "description": "What to change or repair."}},
                "required": ["instruction"], "additionalProperties": False},
            handler=self.modify_input_parameters,
        ))
        registry.register(RegisteredTool(
            name="Run_DRGEP",
            description=("Run mechanism reduction after all DRGEP-required inputs are present and valid. "
                         "The existing reduction workflow always runs three maximum IDT error levels: "
                         "5%, 10%, and 20%. The tool reports each mechanism and its measured metrics."),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=self.run_drgep,
        ))
        registry.register(RegisteredTool(
            name="Ask_question_to_user",
            description="Ask one concise clarification question and return the user's answer.",
            parameters={"type": "object", "properties": {
                "question": {"type": "string"}}, "required": ["question"],
                "additionalProperties": False},
            handler=self.ask_question_to_user,
        ))

    @staticmethod
    def _merge(current: InputParameters, candidate: InputParameters) -> InputParameters:
        data = current.model_dump()
        for field, value in candidate.model_dump().items():
            if value is not None:
                data[field] = value
        return InputParameters.model_validate(data)

    def _validate_and_update(self, candidate: InputParameters, state: AgentState) -> dict[str, Any]:
        normalized, messages = self.validator.validate(candidate)
        state.input_parameters = normalized
        state.validation_messages = messages
        state.validation_errors = self._consistency_errors(normalized)
        state.missing_required_fields = [
            field for field in self.REQUIRED_FOR_DRGEP
            if getattr(normalized, field) is None
        ]
        for field in ("fuel", "pressure_start", "temperature_autoignition_start", "equivalence_ratio_start"):
            if any(field in error for error in state.validation_errors) and field not in state.missing_required_fields:
                state.missing_required_fields.append(field)
        state.status = "ready" if not state.missing_required_fields and not state.validation_errors else "collecting_inputs"
        return {"input_parameters": normalized.model_dump(mode="json"),
                "validation_messages": messages,
                "validation_errors": state.validation_errors,
                "missing_required_fields": state.missing_required_fields,
                "status": state.status}

    @staticmethod
    def _consistency_errors(params: InputParameters) -> list[str]:
        errors: list[str] = []
        if params.fuel:
            fractions = [component.fraction for component in params.fuel]
            if any(value is None or value < 0 or value > 1 for value in fractions):
                errors.append("fuel fractions must all be present and between 0 and 1")
            elif abs(sum(fractions) - 1.0) > 1e-6:
                errors.append(f"fuel fractions must sum to 1 (got {sum(fractions):.6g})")
        ranges = [
            ("pressure_start", params.pressure_start, "pressure_end", params.pressure_end),
            ("temperature_autoignition_start", params.temperature_autoignition_start,
             "temperature_autoignition_end", params.temperature_autoignition_end),
            ("equivalence_ratio_start", params.equivalence_ratio_start,
             "equivalence_ratio_end", params.equivalence_ratio_end),
        ]
        for lower_name, lower, upper_name, upper in ranges:
            if lower is not None and upper is not None and lower > upper:
                errors.append(f"{lower_name} must be less than or equal to {upper_name}")
        if any(value is not None and value <= 0
               for value in (params.pressure_start, params.pressure_end)):
            errors.append("pressure_start and pressure_end must be positive")
        if any(value is not None and value <= 0 for value in (
                params.temperature_autoignition_start, params.temperature_autoignition_end)):
            errors.append("temperature_autoignition_start and temperature_autoignition_end must be positive")
        if ((params.equivalence_ratio_start is not None and params.equivalence_ratio_start <= 0)
                or (params.equivalence_ratio_end is not None and params.equivalence_ratio_end <= 0)):
            errors.append("equivalence_ratio_start and equivalence_ratio_end must be positive")
        return errors

    def retrieve_input_parameters(self, arguments: dict[str, Any], state: AgentState) -> dict[str, Any]:
        schema = InputParameters.model_json_schema()
        spec = self.models.specs.get("parameter_extractor", {})
        prompt_path = spec.get("system_prompt_path")
        if prompt_path and Path(prompt_path).is_file():
            system_prompt = Path(prompt_path).read_text(encoding="utf-8").rstrip()
            prompt = ("Extract parameters from the latest user message. Use recent conversation only "
                      "to interpret a short answer to a clarification question. Return the trained "
                      "JSON format and do not change established values unless the user explicitly "
                      "corrects them.\nRECENT CONVERSATION:\n" +
                      json.dumps([m.model_dump() for m in state.conversation_history[-6:]], ensure_ascii=False) +
                      "\nCURRENT PARAMETERS (preserve fields not addressed by the latest message):\n" +
                      state.input_parameters.model_dump_json(indent=2) +
                      "\nLATEST MESSAGE:\n" + state.user_message)
        else:
            system_prompt = "You extract structured combustion simulation inputs."
            prompt = ("Extract only information explicitly present in the latest user message. "
                      "Return exactly a JSON object with the single top-level key input_parameters. "
                      "Its value must follow this schema; use null for absent values. Do not infer "
                      "values or change units. If multiple fuel species are stated without fractions, "
                      "equal fractions may be used.\nSCHEMA:\n" + json.dumps(schema) +
                      "\nRECENT CONVERSATION (for interpreting short answers to clarification questions):\n" +
                      json.dumps([m.model_dump() for m in state.conversation_history[-6:]], ensure_ascii=False) +
                      "\nLATEST MESSAGE:\n" + state.user_message)
        reply = self.models.generate("parameter_extractor", [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}], max_new_tokens=1200, do_sample=False)
        data = _json_from_reply(reply)
        candidate = InputParameters.model_validate(data.get("input_parameters", data))
        merged = self._merge(state.input_parameters, candidate)
        result = self._validate_and_update(merged, state)
        result["extractor_reply"] = reply
        return result

    def modify_input_parameters(self, arguments: dict[str, Any], state: AgentState) -> dict[str, Any]:
        instruction = arguments["instruction"]
        prompt = ("Update the provided InputParameters only where the instruction calls for a change. "
                  "Preserve all other fields. Do not invent missing values. Return one JSON object "
                  "with the complete InputParameters object as the input_parameters value.\n"
                  "CURRENT INPUT PARAMETERS:\n" + state.input_parameters.model_dump_json(indent=2) +
                  "\nINSTRUCTION:\n" + instruction)
        reply = self.models.generate("parameter_editor", [
            {"role": "system", "content": "You precisely edit structured combustion parameters."},
            {"role": "user", "content": prompt}], max_new_tokens=1600, do_sample=False)
        data = _json_from_reply(reply)
        candidate = InputParameters.model_validate(data.get("input_parameters", data))
        result = self._validate_and_update(self._merge(state.input_parameters, candidate), state)
        result["editor_reply"] = reply
        return result

    @staticmethod
    def _load_metrics(mechanism_name: str) -> dict[str, Any]:
        path = Path("outputs/reduced_mechanisms/metrics") / f"{mechanism_name}-metrics.pkl"
        if not path.is_file():
            return {"mechanism": mechanism_name, "metrics_error": f"Metrics file not found: {path}"}
        with path.open("rb") as handle:
            metrics = pickle.load(handle)
        return _json_safe({"mechanism": mechanism_name, **metrics})

    def run_drgep(self, arguments: dict[str, Any], state: AgentState) -> dict[str, Any]:
        # Recheck even when the orchestrator says the state is ready.
        validation = self._validate_and_update(state.input_parameters, state)
        if validation["missing_required_fields"] or validation["validation_errors"]:
            return {"error": "DRGEP cannot run until required parameters are present and valid.", **validation}
        state.status = "running"
        mechanism_names = AgentToolMechReduction().run_drgep(state.input_parameters)
        results = [self._load_metrics(name) for name in mechanism_names]
        state.reduction_results = results
        if results and all("metrics_error" not in result for result in results):
            # With no user criteria, favor the lowest IDT error, then smaller mechanisms.
            ranked = sorted(results, key=lambda m: (
                m.get("maximum_idt_error_percent", float("inf")),
                m.get("remaining_species_percent", float("inf")),
                m.get("remaining_reactions_percent", float("inf"))))
            state.recommended_mechanism = ranked[0]["mechanism"]
        state.status = "completed"
        return {"error_levels_percent": [5, 10, 20], "mechanisms": results,
                "recommended_mechanism": state.recommended_mechanism,
                "recommendation_rule": "lowest measured IDT error, then species and reaction percentages"}

    @staticmethod
    def ask_question_to_user(arguments: dict[str, Any], state: AgentState) -> dict[str, Any]:
        question = arguments["question"].strip()
        if not question:
            raise ValueError("Question cannot be empty")
        state.pending_question = question
        return {"question": question, "awaiting_user": True}
