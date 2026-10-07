import sys

from .graph import AgentInputGraph
from .llms import ConversationLLM, RetrievalLLM, UpdateLLM, FillInputLLM, RouterLLM, FillCriteriaLLM, SelectMechLLM, RefineParametersLLM
from CombustionAgent.prompts import get_chat_prompt, get_fill_input_prompt, get_retrieve_prompt, get_router_prompt, get_update_prompt, get_fill_criteria_prompt, get_select_mechanism_prompt, get_refine_parameters_prompt
from .model_manager import ModelManager
from .project_manager import ProjectManager
from .parameters import InputParameters, CriteriaParameters, RefinementResult
from .agent_tool import AgentToolMechReduction
from .console import console, logger
from rich.markdown import Markdown
from rich.panel import Panel
import questionary

class Agent:

    def __init__(self, model_name: str, model_opening_message: str, model_opening_message_criteria: str, model_opening_message_iteration: str, database_path: str, project_path: str) -> None:

        self.project_manager = ProjectManager(project_path)
        self.model_manager = ModelManager(model_name)

        schema_input = InputParameters.model_json_schema()
        schema_criteria = CriteriaParameters.model_json_schema()
        schema_refinement = RefinementResult.model_json_schema()
        
        self.LLM_conversation = ConversationLLM(self.model_manager, get_chat_prompt(), model_opening_message)
        self.LLM_retrieval = RetrievalLLM(self.model_manager, get_retrieve_prompt(schema_input, database_path))
        self.LLM_update = UpdateLLM(self.model_manager, get_update_prompt(schema_input))
        self.LLM_fill_input = FillInputLLM(self.model_manager, get_fill_input_prompt(schema_input), database_path)
        self.LLM_router = RouterLLM(self.model_manager, get_router_prompt())
        self.LLM_fill_criteria = FillCriteriaLLM(self.model_manager, get_fill_criteria_prompt(schema_criteria))
        self.LLM_select_mechanism = SelectMechLLM(self.model_manager, get_select_mechanism_prompt())
        self.LLM_refine_parameters = RefineParametersLLM(self.model_manager, get_refine_parameters_prompt(schema_refinement))

        self.model_opening_message = model_opening_message
        self.model_opening_message_criteria = model_opening_message_criteria
        self.model_opening_message_iteration = model_opening_message_iteration

        self.input_graph = AgentInputGraph(self, database_path)

    def main(self) -> None:

        selected_project_ID, new_project = self.project_manager.select_project()

        if(new_project):
            self.workflow_first_interaction()

        else:
            self.workflow_iteration()

    def workflow_first_interaction(self) -> None:

        input_parameters, process_history_input, message_history_input, behind_the_scene_history_input = self.run_input_graph()

        logger.debug(f"Input parameters: {input_parameters}")
        logger.debug(f"Input process history:\n{process_history_input}")

        criteria_parameters, process_history_criteria, message_history_criteria, behind_the_scene_history_criteria = self.run_criteria_graph()

        logger.debug(f"Criteria parameters: {criteria_parameters}")
        logger.debug(f"Criteria process history:\n{process_history_criteria}")

        list_mechanisms = self.run_mechanism_reduction(input_parameters)
        list_mechanisms_metrics_json = self.LLM_select_mechanism.get_mechanism_metrics_json(list_mechanisms)

        reply_selection_mechanism = self.select_mechanism(criteria_parameters, list_mechanisms)

        messages_history = message_history_input + message_history_criteria
        behind_the_scene_history = behind_the_scene_history_input + behind_the_scene_history_criteria

        self.project_manager.save_data(messages_history, behind_the_scene_history, input_parameters, criteria_parameters, list_mechanisms_metrics_json, reply_selection_mechanism)

        console.print(
            Panel(
                Markdown(reply_selection_mechanism),
                title="Agent",
                border_style="cyan"
            )
        )

        # how to handle default ranges and steps in ranges, and default species?

        self.model_manager.unload_model()
        sys.exit()

    def workflow_iteration(self) -> None:

        # Present message with what has been done previous time
        console.print(
            Panel(
                self.model_opening_message_iteration,
                title="Agent",
                border_style="cyan"
            )
        )

        # Get input of user
        user_input = questionary.text("You:").ask()
        
        if user_input is None or user_input.lower() in ["exit", "quit"]:
            self.model_manager.unload_model()
            sys.exit()

        # Understand user message and change input parameters and/or criteria parameters
        previous_summaries = self.project_manager.get_previous_summaries()
        reply_refine_parameters = self.LLM_refine_parameters.refine_parameters(user_input, previous_summaries) #ADD REQUIRED INPUTS
        logger.debug(f"LLM refine parameters reply:\n{reply_refine_parameters}")

        diagnosis = reply_refine_parameters.diagnosis
        reasoning = reply_refine_parameters.reasoning
        input_parameters = reply_refine_parameters.input_parameters
        criteria_parameters = reply_refine_parameters.criteria_parameters

        # Run DRGEP again

        list_mechanisms = self.run_mechanism_reduction(input_parameters)

        # Select best mechanism
        reply_selection_mechanism = self.select_mechanism(criteria_parameters, list_mechanisms) # Add extra context for the selection?
        
        ########################################################################################################
        list_mechanisms_metrics_json = self.LLM_select_mechanism.get_mechanism_metrics_json(list_mechanisms)

        messages_history = [user_input]
        behind_the_scene_history = [diagnosis, reasoning]

        self.project_manager.save_data(messages_history, behind_the_scene_history, input_parameters, criteria_parameters, list_mechanisms_metrics_json, reply_selection_mechanism)
        #######################################################################################################

        # Provide history of what has been done since then
        console.print(
        Panel(
            Markdown(reply_selection_mechanism),
            title="Agent",
            border_style="cyan"
        )
    )

        # Check if better than previous mechanism

        # Save all history ......

        self.model_manager.unload_model()
        sys.exit()


    def run_input_graph(self) -> tuple[InputParameters, list[str]]:

        # complete process of input graph

        state = {
                "user_message": "",
                "process_history": [],
                "working_history": [],
                "message_history": [],
                "behind_the_scene_history": [],
                "input_parameters": InputParameters(),
                "route": None,
                "response": None,
            }

        console.print(
            Panel(
                self.model_opening_message,
                title="Agent",
                border_style="cyan"
            )
        )
        state["process_history"].append(f"Agent: {self.model_opening_message}")
        state["message_history"].append(f"Agent: {self.model_opening_message}")

        while True:
            try:
                user_input = questionary.text("You:").ask()

                if user_input is None or user_input.lower() in ["exit", "quit"]:
                    self.model_manager.unload_model()
                    sys.exit()

                # Reset fields for new cycle
                state["working_history"] = []
                state["route"] = None

                # Update only the part of the state that changes
                state["user_message"] = user_input

                state["process_history"].append(f"User: {user_input}")
                state["message_history"].append(f"User: {user_input}")

                # Run the LangGraph
                state = self.input_graph.app.invoke(state)

                if state["route"] == "END":
                    console.print("[bold green]Agent found the required input parameters.[/bold green]")
                    break

                # Display the response generated by the chat node
                console.print(
                    Panel(
                        Markdown(state["response"]),
                        title="Agent",
                        border_style="cyan"
                    )
                )

            except KeyboardInterrupt:
                self.model_manager.unload_model()
                sys.exit()

        return state["input_parameters"], state["process_history"], state["message_history"], state["behind_the_scene_history"]

    def run_criteria_graph(self) -> tuple[CriteriaParameters, list[str]]:

        process_history = []
        message_history = []
        behind_the_scene_history = []

        console.print(
            Panel(
                self.model_opening_message_criteria,
                title="Agent",
                border_style="cyan"
            )
        )
        process_history.append(f"Agent: {self.model_opening_message_criteria}")
        message_history.append(f"Agent: {self.model_opening_message_criteria}")
 
        try:
            user_input = questionary.text("You:").ask()
            if user_input is None or user_input.lower() in ["exit", "quit"]:
                self.model_manager.unload_model()
                sys.exit()

            process_history.append(f"User: {user_input}")

            console.print("[bold cyan]Agent is retrieving the criteria parameters.[/bold cyan]")

            LLM_fill_criteria_reply, criteria_parameters = self.LLM_fill_criteria.fill_criteria(user_input)

            process_history.append(f"FILL CRITERIA PARAMETERS: the agent has filled the criteria parameters and has assigned the weights as follows: {criteria_parameters}")
            behind_the_scene_history.append(f"FILL CRITERIA PARAMETERS: the agent has filled the criteria parameters and has assigned the weights as follows: {criteria_parameters}")

        except KeyboardInterrupt:
            self.model_manager.unload_model()
            sys.exit()

        return criteria_parameters, process_history, process_history, behind_the_scene_history


    def run_mechanism_reduction(self, input_parameters: InputParameters) -> list[str]:

        console.print("[bold]Launch DRGEP...[/bold]")

        agent_tool_mechanism_reduction = AgentToolMechReduction()
        
        list_mechanisms = agent_tool_mechanism_reduction.run_drgep(input_parameters)
        logger.debug(f"List of generated mechanisms: {list_mechanisms}")

        return list_mechanisms

    def select_mechanism(self, criteria_paramters: CriteriaParameters, list_mechanisms: list[str]) -> str:

        # Provide mechanism + explanation

        console.print("[bold cyan]Agent is selecting the most suitable mechanism.[/bold cyan]")

        reply_selected_mechanism = self.LLM_select_mechanism.select_mechanism(criteria_paramters, list_mechanisms)

        return reply_selected_mechanism