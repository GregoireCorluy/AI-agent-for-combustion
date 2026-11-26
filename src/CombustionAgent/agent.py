"""Tool-using conversational orchestrator for combustion mechanism reduction."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import questionary
from rich.markdown import Markdown
from rich.panel import Panel

from .console import console, logger
from .model_manager import ModelManager
from .project_manager import ProjectManager
from .state import AgentState, ChatMessage, ToolEvent
from .tools import CombustionTools, ToolCall, ToolRegistry, _json_from_reply


ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator for a combustion mechanism reduction assistant.
Choose the next action from the registered tools, or provide a user-facing final response.
Use tools for parameter extraction, parameter updates, deterministic validation, and DRGEP.
Never invent input values, mechanism names, tool results, or scientific metrics. Ask the user
for any missing or unclear input before calling Run_DRGEP. Do not call Run_DRGEP until the
state says it is ready and missing_required_fields is empty. The DRGEP tool always runs three
maximum ignition-delay error levels: 5%, 10%, and 20%. Once the tool returns results, recommend
the best candidate using the measured metrics and explain the trade-off. With no selection
criteria supplied, the tool's ranking rule favors lowest IDT error, then smaller species and
reaction percentages. If the user's latest message corrects established values, use
modify_input_parameters. Keep questions concise and ask only what is needed to continue.

Return exactly one JSON object in one of these forms:
{"action":"tool_call","tool":"registered_tool_name","arguments":{...}}
{"action":"final","response":"user-facing message"}
"""


def default_model_specs() -> dict[str, dict[str, Any]]:
    """Role configuration; environment variables allow replacing or extending models."""
    orchestrator = os.environ.get("COMBUSTION_ORCHESTRATOR_MODEL", "Qwen/Qwen3.8-27B")
    extractor = os.environ.get("COMBUSTION_EXTRACTOR_MODEL", "models/Qwen3-8B")
    adapter = os.environ.get("COMBUSTION_EXTRACTOR_ADAPTER")
    if not adapter:
        latest = Path("fine-tuning/outputs/qwen3-8b-qlora/latest_run.json")
        if latest.is_file():
            try:
                run_name = json.loads(latest.read_text(encoding="utf-8"))["run_name"]
                candidate = Path("fine-tuning/outputs/qwen3-8b-qlora/runs") / run_name / "adapter"
                if candidate.is_dir():
                    adapter = str(candidate)
            except (OSError, KeyError, json.JSONDecodeError):
                logger.warning("Could not resolve the latest parameter-extractor adapter")
    if not adapter:
        logger.warning(
            "No QLoRA adapter was found for the parameter extractor; the configured base model "
            "will be used. Set COMBUSTION_EXTRACTOR_ADAPTER to a trained adapter directory."
        )
    extractor_system_prompt = None
    if adapter:
        saved_prompt = Path(adapter) / "system_prompt.txt"
        if saved_prompt.is_file():
            extractor_system_prompt = str(saved_prompt)
    specs: dict[str, dict[str, Any]] = {
        "orchestrator": {"model": orchestrator, "multimodal": "Qwen3.8" in orchestrator,
                         "quantization": "4bit", "compute_dtype": "bfloat16"},
        "parameter_extractor": {"model": extractor, "adapter": adapter,
                                "system_prompt_path": extractor_system_prompt,
                                "quantization": "4bit", "compute_dtype": "bfloat16"},
        # Parameter editing can use the extractor model until a separate editor is configured.
        "parameter_editor": {"alias_of": "parameter_extractor"},
    }
    editor = os.environ.get("COMBUSTION_EDITOR_MODEL")
    if editor:
        specs["parameter_editor"] = {"model": editor, "quantization": "4bit",
                                     "compute_dtype": "bfloat16"}
    return specs


class Agent:
    """A bounded tool loop. LangGraph is not needed for this linear control pattern."""

    def __init__(self, model_name: str | None = None, model_opening_message: str = "",
                 model_opening_message_criteria: str = "", model_opening_message_iteration: str = "",
                 database_path: str = "data/database/2026-09-14-database-mechanisms-H2-NH3-preliminary.json",
                 project_path: str = "projects/", model_specs: dict[str, dict[str, Any]] | None = None,
                 max_tool_steps: int = 12):
        self.database_path = database_path
        self.max_tool_steps = max_tool_steps
        specs = model_specs or default_model_specs()
        if model_name:
            specs = {**specs, "orchestrator": {**specs.get("orchestrator", {}), "model": model_name,
                                                "multimodal": "qwen3.8" in model_name.lower()}}
        # Eager loading makes resource use and startup failures visible before a conversation begins.
        self.model_manager = ModelManager(specs)
        self.project_manager = ProjectManager(project_path)
        self.registry = ToolRegistry()
        CombustionTools(self.model_manager, database_path).register(self.registry)
        self.opening_message = model_opening_message or (
            "Describe the combustion simulation or mechanism reduction you want to perform."
        )

    def main(self) -> None:
        project_id, new_project = self.project_manager.select_project()
        if not project_id:
            self.model_manager.unload_model()
            return
        state = self._load_state(project_id, new_project)
        if not state.conversation_history:
            self._show(self.opening_message)
            state.conversation_history.append(ChatMessage(role="assistant", content=self.opening_message))
            self._save_state(state)

        try:
            while True:
                user_input = questionary.text("You:").ask()
                if user_input is None or user_input.strip().lower() in {"exit", "quit"}:
                    break
                state.user_message = user_input.strip()
                state.pending_question = None
                state.conversation_history.append(ChatMessage(role="user", content=state.user_message))
                self._save_state(state)
                self._run_turn(state)
        except KeyboardInterrupt:
            console.print("\n[yellow]Conversation interrupted.[/yellow]")
        finally:
            self.model_manager.unload_model()

    def _run_turn(self, state: AgentState) -> None:
        for step in range(self.max_tool_steps):
            reply = self.model_manager.generate(
                "orchestrator", self._orchestrator_messages(state),
                max_new_tokens=1800, do_sample=False, enable_thinking=True,
            )
            try:
                decision = ToolCall.model_validate(_json_from_reply(reply))
            except Exception as exc:
                logger.exception("Could not parse orchestrator response")
                self._show(f"I could not interpret my next action safely ({exc}). Please rephrase your request.")
                state.status = "error"
                self._save_state(state)
                return

            if decision.action == "final":
                answer = (decision.response or "I could not complete the request.").strip()
                self._show(answer)
                state.conversation_history.append(ChatMessage(role="assistant", content=answer))
                self._save_state(state)
                return
            if decision.action != "tool_call" or not decision.tool:
                self._show("I could not determine a valid next step. Please clarify the request.")
                state.status = "error"
                self._save_state(state)
                return

            tool = self.registry.get(decision.tool)
            console.print(f"[bold cyan]Running {tool.name}...[/bold cyan]")
            try:
                result = self.registry.call(tool.name, decision.arguments, state)
            except Exception as exc:
                logger.exception("Tool %s failed", tool.name)
                result = {"error": f"{type(exc).__name__}: {exc}"}
                state.status = "error"
            state.tool_history.append(ToolEvent(tool=tool.name, arguments=decision.arguments, result=result))
            self._save_state(state)

            if tool.name == "Ask_question_to_user":
                answer = questionary.text(result["question"]).ask()
                if answer is None:
                    return
                state.user_message = answer.strip()
                state.pending_question = None
                state.conversation_history.append(ChatMessage(role="assistant", content=result["question"]))
                state.conversation_history.append(ChatMessage(role="user", content=state.user_message))
                self._save_state(state)
        else:
            message = "I reached the action limit for this turn. Please tell me how you would like to continue."
            self._show(message)
            state.conversation_history.append(ChatMessage(role="assistant", content=message))
            self._save_state(state)

    def _orchestrator_messages(self, state: AgentState) -> list[dict[str, str]]:
        system = ORCHESTRATOR_SYSTEM_PROMPT + "\nAVAILABLE TOOLS:\n" + json.dumps(
            self.registry.descriptions(), ensure_ascii=False, indent=2
        ) + "\nCURRENT STATE (authoritative):\n" + json.dumps(
            state.orchestrator_view(), ensure_ascii=False, indent=2
        )
        messages = [{"role": "system", "content": system}]
        # State already contains compact recent history and latest tool outcomes.
        for message in state.conversation_history[-8:]:
            messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": "Choose the next tool or final response for the current task."})
        return messages

    def _load_state(self, project_id: str, new_project: bool) -> AgentState:
        path = Path(self.project_manager.selected_project_path) / "session_state.json"
        if path.is_file():
            state = AgentState.model_validate_json(path.read_text(encoding="utf-8"))
            state.project_id = project_id
            return state
        state = AgentState(project_id=project_id)
        if not new_project:
            summaries = self.project_manager.get_previous_summaries()
            if summaries:
                saved = summaries[-1].get("input_parameters") or {}
                state.input_parameters = state.input_parameters.model_validate(saved)
                state.missing_required_fields = [
                    field for field in CombustionTools.REQUIRED_FOR_DRGEP
                    if getattr(state.input_parameters, field) is None
                ]
                state.status = "ready" if not state.missing_required_fields else "collecting_inputs"
                state.conversation_history.append(ChatMessage(
                    role="assistant", content="I restored the input parameters from the last saved project iteration."
                ))
        return state

    def _save_state(self, state: AgentState) -> None:
        if self.project_manager.selected_project_path is None:
            return
        path = Path(self.project_manager.selected_project_path) / "session_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _show(message: str) -> None:
        console.print(Panel(Markdown(message), title="Agent", border_style="cyan"))
