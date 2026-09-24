from .state import AgentState
from langgraph.graph import StateGraph, START, END
from .parameters import InputParameters
import re
import unicodedata
from rapidfuzz import process, fuzz
from .database import MechanismDatabase
import cantera as ct

class AgentGraph:

    def __init__(self, agent, database_path):

        self.agent = agent
        self.database_path = database_path

        self.graph = StateGraph(AgentState)

        self.graph.add_node("router", self.router_node)
        self.graph.add_node("chat", self.chat_node)
        self.graph.add_node("retrieve", self.retrieve_node)
        #self.graph.add_node("verify", self.verify_node)
        self.graph.add_node("update", self.update_node)
        self.graph.add_node("fill", self.fill_node)

        self.graph.add_edge(START, "router")
        self.graph.add_conditional_edges(
                                    "router",
                                    self.route_after_router,
                                    {
                                        "CHAT": "chat",
                                        "RETRIEVE": "retrieve",
                                        "UPDATE": "update",
                                        "END": END,
                                    }
                                )
        self.graph.add_edge("chat", END) # End the graph at the end of every iteration, waiting on the answer of the user
        self.graph.add_edge("fill", "chat")
        self.graph.add_edge("update", "chat")
        self.graph.add_conditional_edges(
                                    "retrieve",
                                    self.route_after_retrieve,
                                    {
                                        "chat": "chat",
                                        "fill": "fill",
                                    }
                                )

        self.app = self.graph.compile()

    def router_node(self, state: AgentState):

        possible_actions = ["CHAT"]

        if all(value is not None  for field, value in state["input_parameters"].model_dump().items() if field != "application_regime"):
            possible_actions.append("END")
            possible_actions.append("UPDATE")
        elif all(value is None for value in state["input_parameters"].model_dump().values()):
            possible_actions.append("RETRIEVE")
        else:
            possible_actions.append("UPDATE") #should theoretically not exist since all the fields should be filled in after the retrieve

        selected_route = self.agent.LLM_router.define_route(state["response"], state["user_message"], state["input_parameters"], possible_actions)

        print(f"Selected route: {selected_route}")

        return {"route": selected_route}

    def chat_node(self, state: AgentState):

        message = f"""
                    CURRENT USER MESSAGE:
                    {state["user_message"]}

                    PROCESS HISTORY:
                    {state["process_history"]}

                    CURRENT INPUT PARAMETERS:
                    {state["input_parameters"]}

                    TASK:
                    Respond naturally and coherently to the user's latest message.
                    Take into account the process history, which summarizes what the agent has done since the last user's message, and the current input
                    parameters.

                    If further information is required, ask the user for it.
                    Do not invent information.
                    """

        response = self.agent.LLM_conversation.generate(message)

        return {
            "response": response
        }

    def retrieve_node(self, state: AgentState):

        history_entries = []

        LLM_retrieval_reply, input_parameters = self.agent.LLM_retrieval.retrieve_information(state["user_message"])

        print(f"\nAgent has retrieved from the user input: {LLM_retrieval_reply}")
        print(f"Input parameters from LLM_retrieval: {input_parameters}")

        # Provide the information of which input parameters retrieved
        history_entries.extend([f"Retrieved parameters from the user input: {input_parameters}"])

        # LLM_verification_reply = self.agent.LLM_verification.verify_information(state["user_message"], input_parameters)

        # print(f"\nVerification by the agent: {LLM_verification_reply}")

        # LLM_update_reply, input_parameters_updated = self.agent.LLM_update.update_information(LLM_verification_reply, input_parameters)
        
        # print(f"\nUpdate by the agent: {LLM_update_reply}")
        # print(f"Input parameters after LLM_update: {input_parameters_updated}")

        # Check for consistency in retrieved data

        # Convert the name of the chemical mechanism
        input_parameters, message_mechanism = self.convert_mechanism_name(input_parameters)

        history_entries.extend(message_mechanism)

        # Check for keywords corresponding to regime/application
        database = MechanismDatabase(self.database_path)
        input_parameters.application_regime = match_application_regimes(input_parameters.application_regime, database.get_unique_application_regime())

        # Convert pressure and temperature to bar and Kelvin if necessary
        input_parameters, message_unit = self.convert_units(input_parameters)

        history_entries.extend(message_unit)

        # Standardize for fuel too, check if in the list
        # ...

        # Check for species correspond to mechanism + check species names are chemical species names
        # Combine fuzzy and then convert
        # convert fuel, targeted and retained species names
        input_parameters, message_standardize_species = self.standardize_species(input_parameters)
        history_entries.extend(message_standardize_species)

        # remove duplicates - this is already done in the standardize_species function
        #input_parameters_updated = self.remove_duplicate_species(input_parameters_updated)

        # Remove species which are not in the mechanism
        
        input_parameters, message_species_validation = self.validate_species(input_parameters)
        history_entries.extend(message_species_validation)

        print(f"Input parameters after normalization: {input_parameters}")

        if all(value is not None  for field, value in state["input_parameters"].model_dump().items() if field != "application_regime"):
            history_entry_retrieval = (
                            "RETRIEVAL RESULT: All required input parameters are currently filled. "
                            "The agent should present the extracted parameters to the user and "
                            "ask for confirmation."
                        )
        elif all(value is None for value in input_parameters.model_dump().values()):
            history_entry_retrieval = (
                            "RETRIEVAL RESULT: None of the input parameters have been retrieved from the user's message, all parameters will be inferred by the fill in function."
                        )
        else:
            filled_fields = [
                        field_name
                        for field_name, value in input_parameters.model_dump().items()
                        if value is not None
                    ]
            history_entry_retrieval = (
                            f"RETRIEVAL RESULT: The fields {filled_fields} of the input parameters have been retrieved from the user's message, the remaining ones will be retrieved by the fill in function."
                        )
            
        history_entries.append(history_entry_retrieval)

        print("total history entries retrieval")
        print(history_entries)

        return {"input_parameters": input_parameters,
                "process_history": state["process_history"] + history_entries}

    # def verify_node(self, state: AgentState):
    #     result = self.agent.verify(...)
    #     return {...}

    def update_node(self, state: AgentState):

        LLM_reply, input_parameters_filled = self.agent.LLM_update.update_information(state["user_message"], state["input_parameters"])
        
        print(f"\nAgent: {LLM_reply}")

        history_entry = (
                        "UPDATE RESULT: The input parameters have been updated according to the user's request. The agent should present the extracted parameters to the user and ask for confirmation."
                    )

        return {"input_parameters": input_parameters_filled,
                "process_history": state["process_history"] + [history_entry]}

    def fill_node(self, state: AgentState):

        history_entries = []

        LLM_fill_reply, input_parameters_filled, matched_results = self.agent.LLM_fill.fill_missing_information(state["input_parameters"])
        
        print(f"\nAgent has filled in the missing fields using the database: {LLM_fill_reply}")
        print(f"Input parameters after LLM_fill: {input_parameters_filled}")

        previous = state["input_parameters"].model_dump()
        filled = input_parameters_filled.model_dump()

        newly_filled_fields = [
            field_name
            for field_name, old_value in previous.items()
            if old_value is None and filled[field_name] is not None
        ]

        # Remove duplicate species
        input_parameters_filled = self.remove_duplicate_species(input_parameters_filled)
        print(f"Input parameters after removal duplicate species: {input_parameters_filled}")

        # Function to reset the values from the user?
        # ... (Now the agent sets the values based on the database)

        # Remove species which are not in the mechanism
        # And standardize species names?
        input_parameters_filled, message_species_validation = self.validate_species(input_parameters_filled)
        history_entries.extend(message_species_validation)

        print(f"Input parameters after removal non-existant species: {input_parameters_filled}")

        matched_cases_ID = [case.get("id") for case in matched_results]

        history_entry_fill = (
                        f"FILL RESULT: The fields {newly_filled_fields} of the input parameters, that were missing from the user's message, have been filled based on the context provided by the user and combined with the retrieval from a combustion database. The fields {newly_filled_fields} were filled based on cases with ID {matched_cases_ID} from the combustion database. The agent should present the extracted parameters to the user and ask for confirmation."
                    )

        history_entries.append(history_entry_fill)

        return {"input_parameters": input_parameters_filled,
                "process_history": state["process_history"] + history_entries}
    
    def route_after_router(self, state: AgentState):
        return state["route"]

    def route_after_retrieve(self, state: AgentState):
        if all(value is not None  for field, value in state["input_parameters"].model_dump().items() if field != "application_regime"):
            return "chat" #maybe don't use the chat in that case but directly print it? How to format the string in the history for the LLM
        
        return "fill"

    def remove_duplicate_species(self, input_parameters: InputParameters,
                                ) -> InputParameters:

        if input_parameters.retained_species:
            input_parameters.retained_species = list(
                dict.fromkeys(input_parameters.retained_species)
            )

        if input_parameters.target_species:
            input_parameters.target_species = list(
                dict.fromkeys(input_parameters.target_species)
            )

        return input_parameters

    # Did not consider the case where the list was currently None and which needed to stay None
    # def validate_species(self, input_parameters: InputParameters
    #                     ) -> tuple[InputParameters, list[str]]:

    #     messages = []

    #     path = "data/mechanisms/detailed/"
    #     gas = ct.Solution(path + input_parameters.mechanism + ".yaml")
    #     mechanism_species = set(gas.species_names)

    #     retained_species = input_parameters.retained_species or []
    #     target_species = input_parameters.target_species or []

    #     valid_retained = [
    #         species
    #         for species in retained_species
    #         if species in mechanism_species
    #     ]

    #     valid_target = [
    #         species
    #         for species in target_species
    #         if species in mechanism_species
    #     ]

    #     removed_retained = [
    #         species
    #         for species in retained_species
    #         if species not in mechanism_species
    #     ]

    #     removed_target = [
    #         species
    #         for species in target_species
    #         if species not in mechanism_species
    #     ]

    #     input_parameters.retained_species = valid_retained
    #     input_parameters.target_species = valid_target

    #     if removed_retained:
    #         messages.append(
    #             f"Removed retained species not present in mechanism: "
    #             f"{removed_retained}"
    #         )

    #     if removed_target:
    #         messages.append(
    #             f"Removed target species not present in mechanism: "
    #             f"{removed_target}"
    #         )

    #     return input_parameters, messages

    def validate_species(
                            self,
                            input_parameters: InputParameters,
                        ) -> tuple[InputParameters, list[str]]:

        messages = []

        if input_parameters.mechanism is None:
            return input_parameters, messages

        path = "data/mechanisms/detailed/"
        gas = ct.Solution(path + input_parameters.mechanism + ".yaml")
        mechanism_species = set(gas.species_names)

        for field_name in ["retained_species", "target_species"]:

            species_list = getattr(input_parameters, field_name)

            # Keep None as None
            if species_list is None:
                continue

            valid_species = []
            removed_species = []

            for species in species_list:

                if species in mechanism_species:
                    valid_species.append(species)
                else:
                    removed_species.append(species)

            # Remove duplicates while preserving order
            valid_species = list(dict.fromkeys(valid_species))

            setattr(
                input_parameters,
                field_name,
                valid_species if valid_species else None,
            )

            if removed_species:
                messages.append(
                    f"Removed species from {field_name.replace('_', ' ')} because not present "
                    f"in mechanism: {list(dict.fromkeys(removed_species))}"
                )

        return input_parameters, messages

    def convert_units(self, input_parameters: InputParameters):

        messages = []

        # -------------------------
        # Temperature
        # -------------------------
        if (
            input_parameters.temperature_start is not None
            or input_parameters.temperature_end is not None
        ):

            if input_parameters.temperature_unit is None:
                messages.append(
                    "Temperature unit is missing. The agent must ask the user "
                    "to specify the temperature unit before continuing."
                )

            else:
                unit = input_parameters.temperature_unit.strip().lower()

                temperature_unit_aliases = {
                    # Kelvin
                    "k": "k",
                    "kelvin": "k",
                    "kelvins": "k",
                    "°k": "k",

                    # Celsius
                    "c": "c",
                    "°c": "c",
                    "celsius": "c",
                    "degrees celsius": "c",

                    # Fahrenheit
                    "f": "f",
                    "°f": "f",
                    "fahrenheit": "f",
                    "degrees fahrenheit": "f",
                }

                unit = temperature_unit_aliases.get(unit)

                temperature_conversions = {
                    "k":  lambda x: x,
                    "c":  lambda x: x + 273.15,
                    "f":  lambda x: (x - 32) * 5 / 9 + 273.15,

                }

                if unit not in temperature_conversions:
                    messages.append(
                        f"Unknown temperature unit '{input_parameters.temperature_unit}'. "
                        "The agent must ask the user to clarify the temperature unit."
                    )

                    input_parameters.temperature_unit = None

                elif unit in ["k"]:
                    input_parameters.temperature_unit = "K"

                else:
                    conversion = temperature_conversions[unit]

                    old_start = input_parameters.temperature_start
                    old_end = input_parameters.temperature_end
                    old_unit = input_parameters.temperature_unit

                    if old_start is not None:
                        input_parameters.temperature_start = conversion(old_start)

                    if old_end is not None:
                        input_parameters.temperature_end = conversion(old_end)

                    input_parameters.temperature_unit = "K"

                    messages.append(
                        f"Temperature converted from '{old_unit}' "
                        f"to K: {old_start}-{old_end} {old_unit} to "
                        f"{input_parameters.temperature_start}-"
                        f"{input_parameters.temperature_end} K."
                    )

        # -------------------------
        # Pressure
        # -------------------------
        if (
            input_parameters.pressure_start is not None
            or input_parameters.pressure_end is not None
        ):

            if input_parameters.pressure_unit is None:
                messages.append(
                    "Pressure unit is missing. The agent must ask the user "
                    "to specify the pressure unit before continuing."
                )

            else:
                unit = input_parameters.pressure_unit.strip().lower()

                pressure_unit_aliases = {
                    # Pascal
                    "p": "pa",
                    "pa": "pa",
                    "pascal": "pa",
                    "pascals": "pa",

                    # Kilopascal
                    "kpa": "kpa",
                    "kilo pa": "kpa",
                    "kilopascal": "kpa",
                    "kilopascals": "kpa",
                    "kilo pascal": "kpa",
                    "kilo pascals": "kpa",

                    # Megapascal
                    "mpa": "mpa",
                    "mega pa": "mpa",
                    "megapascal": "mpa",
                    "megapascals": "mpa",
                    "mega pascal": "mpa",
                    "mega pascals": "mpa",

                    # Gigapascal
                    "gpa": "gpa",
                    "giga pa": "gpa",
                    "gigapascal": "gpa",
                    "gigapascals": "gpa",
                    "giga pascal": "gpa",
                    "giga pascals": "gpa",

                    # Bar
                    "bar": "bar",
                    "bars": "bar",

                    # Millibar
                    "milibar": "mbar",
                    "milibars": "mbar",
                    "millibar": "mbar",
                    "millibars": "mbar",
                    "mili bar": "mbar",
                    "mili bars": "mbar",
                    "milli bar": "mbar",
                    "milli bars": "mbar",

                    # Atmosphere
                    "atm": "atm",
                    "atmosphere": "atm",
                    "atmospheres": "atm",

                    # PSI
                    "psi": "psi",
                    "psia": "psi",
                    "psi a": "psi",
                    "psig": "psi",
                    "psi g": "psi",
                    "pounds per square inch": "psi",
                    "pound per square inch": "psi",
                    "pounds per square inch absolute": "psi",
                    "pounds per square inch gauge": "psi",

                    # KSI
                    "ksi": "ksi",
                    "ksia": "ksi",
                    "ksi a": "ksi",
                    "ksig": "ksi",
                    "ksi g": "ksi",
                    "kilopound per square inch": "ksi",
                    "kilopounds per square inch": "ksi",
                    "kip per square inch": "ksi",
                    "kips per square inch": "ksi",

                    # Torr
                    "torr": "torr",
                    "torrs": "torr",

                    # mmHg
                    "mmhg": "mmhg",
                    "mm hg": "mmhg",
                    "mm-hg": "mmhg",
                    "millimeter mercury": "mmhg",
                    "millimeters mercury": "mmhg",
                    "millimetre mercury": "mmhg",
                    "millimetres mercury": "mmhg",
                    "millimeter of mercury": "mmhg",
                    "millimeters of mercury": "mmhg",
                    "millimetre of mercury": "mmhg",
                    "millimetres of mercury": "mmhg",
                }

                unit = pressure_unit_aliases.get(unit)

                pressure_to_bar = {
                    "pa": 1e-5,
                    "kpa": 1e-2,
                    "mpa": 10.0,
                    "gpa": 1e4,

                    "bar": 1.0,
                    "mbar": 1e-3,

                    "atm": 1.01325,

                    "psi": 0.0689475729,
                    "ksi": 68.9475729,

                    "torr": 1.333223684e-3,
                    "mmhg": 1.333223684e-3,
                }

                if unit not in pressure_to_bar:
                    messages.append(
                        f"Unknown pressure unit '{input_parameters.pressure_unit}'. "
                        "The agent must ask the user to clarify the pressure unit."
                    )

                    input_parameters.pressure_unit = None

                elif unit in ["bar"]:
                    input_parameters.pressure_unit = "bar"
                    
                else:
                    factor = pressure_to_bar[unit]

                    old_start = input_parameters.pressure_start
                    old_end = input_parameters.pressure_end
                    old_unit = input_parameters.pressure_unit

                    if old_start is not None:
                        input_parameters.pressure_start *= factor

                    if old_end is not None:
                        input_parameters.pressure_end *= factor

                    input_parameters.pressure_unit = "bar"

                    messages.append(
                        f"Pressure converted from '{old_unit}' to bar: "
                        f"{old_start}-{old_end} {old_unit} to "
                        f"{input_parameters.pressure_start}-"
                        f"{input_parameters.pressure_end} bar."
                    )

        return input_parameters, messages

    def convert_mechanism_name(self, input_parameters):

        messages = []

        # No mechanism provided by the user
        if input_parameters.mechanism is None:
            return input_parameters, messages

        database = MechanismDatabase(self.database_path)
        mechanism_names = database.get_unique_mechanisms()

        user_mechanism = input_parameters.mechanism

        matched_mechanism = match_mechanism_name(
            user_mechanism,
            mechanism_names,
        )

        # A matching mechanism was found
        if matched_mechanism is not None:

            # Only report a conversion if the name was actually changed
            if user_mechanism.strip() != matched_mechanism:
                messages.append(
                    f"MECHANISM: The mechanism name '{user_mechanism}' "
                    f"was matched to the database mechanism '{matched_mechanism}'."
                )

            input_parameters.mechanism = matched_mechanism

        # No matching mechanism was found
        else:

            messages.append(
                f"MECHANISM: The mechanism name '{user_mechanism}' "
                f"could not be recognized in the mechanism database. "
                f"The mechanism parameter has been set to None and must be "
                f"provided again by the user."
            )

            input_parameters.mechanism = None

        return input_parameters, messages

    def standardize_species(self, input_parameters: InputParameters
                            ) -> InputParameters:

        messages = []

        for field_name in ["retained_species", "target_species"]:

            species_list = getattr(input_parameters, field_name)

            if not species_list:
                continue

            standardized_species = []

            for species in species_list:
                result = standardize_species_name(species)

                if result is not None:
                    standardized_species.extend(result)

            # Remove duplicates while preserving order
            standardized_species = list(dict.fromkeys(standardized_species))

            # TO MODIFY: only if standardized species is not None4
            # TO CHECK if correct (when empty)

            messages.extend(f"List of species provided by the user {standardized_species}{' ' if standardized_species else 'not'} recognized for {field_name}")
            setattr(
                input_parameters,
                field_name,
                standardized_species if standardized_species else None,
            )
                 

        return input_parameters, messages

def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.lower()

    # Remove accents
    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
    )

    # Normalize separators
    name = re.sub(r"[-_/]+", " ", name)

    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name

def match_mechanism_name(
                            user_name: str,
                            mechanisms: list[str],
                            threshold: int = 50,
                        ) -> str | None:

    normalized_user = normalize_name(user_name)

    normalized_mechanisms = {
        normalize_name(name): name
        for name in mechanisms
    }

    result = process.extractOne(
        normalized_user,
        normalized_mechanisms.keys(),
        scorer=fuzz.WRatio,
    )

    if result is None:
        return None

    matched_normalized, score, _ = result

    if score >= threshold:
        return normalized_mechanisms[matched_normalized]

    return None

def match_application_regimes(
                                user_keywords: list[str],
                                keywords: list[str],
                                threshold: int = 70,
                            ) -> list[str]:

    normalized_keywords = {
        normalize_name(keyword): keyword
        for keyword in keywords
    }

    if not user_keywords:
        return None

    matched_keywords = []

    for user_keyword in user_keywords:

        normalized_user = normalize_name(user_keyword)

        result = process.extractOne(
            normalized_user,
            normalized_keywords.keys(),
            scorer=fuzz.WRatio,
        )

        if result is None:
            continue

        matched_normalized, score, _ = result

        if score >= threshold:
            matched_keywords.append(
                normalized_keywords[matched_normalized]
            )

    if not matched_keywords:
        return None

    return matched_keywords

SPECIES_ALIASES = {
    "hydrogen": "H2",
    "h2": "H2",

    "oxygen": "O2",
    "o2": "O2",

    "nitrogen": "N2",
    "n2": "N2",

    "hydrogen atom": "H",
    "hydrogen radical": "H",
    "h": "H",

    "oxygen atom": "O",
    "oxygen radical": "O",
    "o": "O",

    "hydroxyl": "OH",
    "hydroxyl radical": "OH",
    "oh": "OH",

    "hydroperoxyl": "HO2",
    "hydroperoxyl radical": "HO2",
    "ho2": "HO2",

    "hydrogen peroxide": "H2O2",
    "h2o2": "H2O2",

    # Water
    "water": "H2O",
    "h2o": "H2O",

    "ammonia": "NH3",
    "nh3": "NH3",

    "amino": "NH2",
    "amino radical": "NH2",
    "nh2": "NH2",

    "amidogen": "NH",
    "amidogen radical": "NH",
    "nh": "NH",

    "nitrogen hydride": "NH",

    "methane": "CH4",
    "ch4": "CH4",

    "carbon monoxide": "CO",
    "co": "CO",

    "carbon dioxide": "CO2",
    "co2": "CO2",

    "nitric oxide": "NO",
    "nitrogen monoxide": "NO",
    "no": "NO",

    "nitrogen dioxide": "NO2",
    "no2": "NO2",

    "nitrous oxide": "N2O",
    "n2o": "N2O",

    "nitrogen atom": "N",
    "nitrogen radical": "N",
    "n": "N",

    "nitrogen trioxide": "NO3",
    "no3": "NO3",

    "nitrogen dioxide radical": "NO2",

    "nitrosyl": "HNO",
    "nitroxyl": "HNO",
    "hno": "HNO",

    "nitrosamine": "NH2NO",
    "nh2no": "NH2NO",
}

SPECIES_GROUPS = {
    "nox": ["NO", "NO2", "N2O"],
    "nitrogen oxides": ["NO", "NO2", "N2O"],

    "nhx": ["NH", "NH2", "NH3"],
    "hydrogen oxides": ["OH", "HO2", "H2O2", "H2O"],
}

def standardize_species_name(
                                species: str,
                                threshold: int = 70,
                            ) -> list[str] | None:

    normalized = normalize_name(species)

    # Exact species alias
    if normalized in SPECIES_ALIASES:
        return [SPECIES_ALIASES[normalized]]

    # Exact species group
    if normalized in SPECIES_GROUPS:
        return SPECIES_GROUPS[normalized].copy()

    # Fuzzy matching against both aliases and groups
    all_names = list(SPECIES_ALIASES) + list(SPECIES_GROUPS)

    result = process.extractOne(
        normalized,
        all_names,
        scorer=fuzz.WRatio,
    )

    if result is None:
        return None

    matched_name, score, _ = result

    if score < threshold:
        return None

    if matched_name in SPECIES_ALIASES:
        return [SPECIES_ALIASES[matched_name]]

    return SPECIES_GROUPS[matched_name].copy()