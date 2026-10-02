import sys

from .graph import AgentInputGraph
from .llms import ConversationLLM, RetrievalLLM, VerifyLLM, UpdateLLM, FillInputLLM, RouterLLM, FillCriteriaLLM, SelectMechLLM, RefineParametersLLM
from .model_manager import ModelManager
from .project_manager import ProjectManager
from .parameters import InputParameters, CriteriaParameters, FuelComponent
from .agent_tool import AgentToolMechReduction
from .console import console, logger
from rich.markdown import Markdown
from rich.panel import Panel
import questionary

class Agent:

    def __init__(self, model_name: str, model_preprompts: list[str], model_opening_message: str, model_opening_message_criteria: str, model_opening_message_iteration: str, database_path: str, project_path: str) -> None:

        self.project_manager = ProjectManager(project_path)
        self.model_manager = ModelManager(model_name)
        
        self.LLM_conversation = ConversationLLM(self.model_manager, model_preprompts[0], model_opening_message)
        self.LLM_retrieval = RetrievalLLM(self.model_manager, model_preprompts[1])
        self.LLM_verification = VerifyLLM(self.model_manager, model_preprompts[2])
        self.LLM_update = UpdateLLM(self.model_manager, model_preprompts[3])
        self.LLM_fill_input = FillInputLLM(self.model_manager, model_preprompts[4], database_path)
        self.LLM_router = RouterLLM(self.model_manager, model_preprompts[5])
        self.LLM_fill_criteria = FillCriteriaLLM(self.model_manager, model_preprompts[6])
        self.LLM_select_mechanism = SelectMechLLM(self.model_manager, model_preprompts[7])
        self.LLM_refine_parameters = RefineParametersLLM(self.model_manager, model_preprompts[8])

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

        # input_parameters, process_history_input, message_history_input, behind_the_scene_history_input = self.run_input_graph()

        # logger.debug(f"Input parameters: {input_parameters}")
        # logger.debug(f"Input process history:\n{process_history_input}")

        # criteria_parameters, process_history_criteria, message_history_criteria, behind_the_scene_history_criteria = self.run_criteria_graph()

        # logger.debug(f"Criteria parameters: {criteria_parameters}")
        # logger.debug(f"Criteria process history:\n{process_history_criteria}")

        #list_mechanisms = self.run_mechanism_reduction(input_parameters)
        list_mechanisms = ["2026-09-29-ID005-Glarborg-2024-NH3-error5", "2026-09-29-ID004-Glarborg-2024-NH3-error5", "2026-09-29-ID003-Glarborg-2024-NH3-error20"]
        list_mechanisms_metrics_json = self.LLM_select_mechanism.get_mechanism_metrics_json(list_mechanisms)

        # reply_selection_mechanism = self.select_mechanism(criteria_parameters, list_mechanisms)

        ########################################################
        # # Add to history?, Use user input?

        input_parameters = InputParameters(
            mechanism="Glarborg-2024-NH3",
            application_regime=None,
            fuel=[
                FuelComponent(species="H2", fraction=1.0)
            ],
            pressure_start=0.5,
            pressure_end=10.0,
            pressure_unit="bar",
            temperature_start=900.0,
            temperature_end=2000.0,
            temperature_unit="K",
            equivalence_ratio_start=0.5,
            equivalence_ratio_end=5.5,
            retained_species=["H2", "N2", "O2"],
            target_species=["H2"],
        )

        criteria_parameters = CriteriaParameters(IDT_accuracy = 1, species_reduction = 0.5, reactions_reduction = 0.5)

        reply_selection_mechanism = "We selected this mechanism for these reasons..."

        ##########################

        messages_history = ["Test"] #message_history_input + message_history_criteria
        behind_the_scene_history = ["Test 2"] #behind_the_scene_history_input + behind_the_scene_history_criteria

        self.project_manager.save_data(messages_history, behind_the_scene_history, input_parameters, criteria_parameters, list_mechanisms_metrics_json, reply_selection_mechanism)

        console.print(
            Panel(
                Markdown(reply_selection_mechanism),
                title="Agent",
                border_style="cyan"
            )
        )

        # Function or script to save all this
        # all messages history
        # input parameters
        # criteria parameters
        # behind the scene history (+range conditions?)
        # list of mechanisms with their metrics
        # Selection: how to handle the choice (exact know which one?)
        # Create summary for the agent

        # one json file for each iteration

        # how to handle default ranges and steps in ranges, and default species?

        self.model_manager.unload_model()
        sys.exit()

    def workflow_iteration(self) -> None:

        # Check empty projects and remove them?

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

        input_parameters, criteria_parameters = self.LLM_refine_parameters.refine_parameters() #ADD REQUIRED INPUTS

        # Run DRGEP again

        input_parameters = None

        list_mechanisms = self.run_mechanism_reduction(input_parameters)

        # Select best mechanism
        criteria_parameters = None
        reply_selection_mechanism = self.select_mechanism(criteria_parameters, list_mechanisms) # Add extra context for the selection?

        ########################################################################################################"
        # 
        # #list_mechanisms = self.run_mechanism_reduction(input_parameters)
        list_mechanisms = ["2026-09-29-ID005-Glarborg-2024-NH3-error5", "2026-09-29-ID004-Glarborg-2024-NH3-error5", "2026-09-29-ID003-Glarborg-2024-NH3-error20"]
        list_mechanisms_metrics_json = self.LLM_select_mechanism.get_mechanism_metrics_json(list_mechanisms)

        # reply_selection_mechanism = self.select_mechanism(criteria_parameters, list_mechanisms)

        ########################################################
        # # Add to history?, Use user input?

        input_parameters = InputParameters(
            mechanism="Glarborg-2024-NH3",
            application_regime=None,
            fuel=[
                FuelComponent(species="H2", fraction=1.0)
            ],
            pressure_start=0.5,
            pressure_end=20.0,
            pressure_unit="bar",
            temperature_start=900.0,
            temperature_end=1500.0,
            temperature_unit="K",
            equivalence_ratio_start=0.5,
            equivalence_ratio_end=1.5,
            retained_species=["H2", "N2", "O2"],
            target_species=["H2"],
        )

        criteria_parameters = CriteriaParameters(IDT_accuracy = 1, species_reduction = 0.5, reactions_reduction = 0.5)

        reply_selection_mechanism = "We selected this new mechanism for these other reasons..."

        ##########################

        messages_history = ["This didn't work"] #message_history_input + message_history_criteria
        behind_the_scene_history = ["We refined the parameters as following"] #behind_the_scene_history_input + behind_the_scene_history_criteria

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

        return None


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