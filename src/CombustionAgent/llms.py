from .models import ModelManager
from .parameters import InputParameters
from .database import MechanismDatabase
import json

class LLM:

    def __init__(self, model_manager: ModelManager, model_preprompt: str) -> None:

        self.model_manager = model_manager
        self.preprompt = model_preprompt
        # set temperature

    def generate(self, message: str, max_new_tokens: int = 200, do_sample: bool = True, enable_thinking = True) -> str:

        messages = [
            {"role": "system", "content": self.preprompt},
            {"role": "user", "content": message},
        ]

        inputs = self.model_manager.tokenizer.apply_chat_template(
                                                messages,
                                                add_generation_prompt=True,
                                                return_tensors="pt",
                                                enable_thinking = enable_thinking #added vs Llama
                                            ).to(self.model_manager.model.device)

        outputs = self.model_manager.model.generate( #add temperature?
                                                **inputs,
                                                max_new_tokens=max_new_tokens,
                                                do_sample=do_sample
                                            )
        # Part added 
        output_ids = outputs[0][inputs["input_ids"].shape[-1]:].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.model_manager.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        LLM_reply = self.model_manager.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # LLM_reply = self.model_manager.tokenizer.decode(
        #                                     outputs[0][inputs["input_ids"].shape[-1]:],
        #                                     skip_special_tokens=True,
        #                                 )

        print(f"Thinking: {thinking_content}")

        return LLM_reply

class ConversationLLM(LLM):

    def __init__(self, model_manager: ModelManager, model_preprompt: str, model_opening_message: str) -> None:

        super().__init__(model_manager, model_preprompt)

        self.history = [{"role": "system", "content": model_preprompt},
                        {"role": "assistant", "content": model_opening_message},]

    def generate(self, message: str, max_new_tokens: int = 5000, do_sample: bool = True) -> str:

        self.history.append({"role": "user", "content": message})

        inputs = self.model_manager.tokenizer.apply_chat_template(
                                self.history,
                                add_generation_prompt=True,
                                return_tensors="pt",
                            ).to(self.model_manager.model.device)

        outputs = self.model_manager.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                            )

        # LLM_reply = self.model_manager.tokenizer.decode(
        #                         outputs[0][inputs["input_ids"].shape[-1]:],
        #                         skip_special_tokens=True,
        #                     )

        output_ids = outputs[0][inputs["input_ids"].shape[-1]:].tolist()
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.model_manager.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        LLM_reply = self.model_manager.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print(f"Thinking: {thinking_content}")

        self.history.append({"role": "assistant", "content": LLM_reply})

        return LLM_reply

class RetrievalLLM(LLM):

    def retrieve_information(self, message: str, max_new_tokens: int = 1000) -> tuple[str, InputParameters]: #previously 300 max new tokens

        #estimation for the uncertain parts?
        #set temperature

        LLM_reply = self.generate(message, max_new_tokens = max_new_tokens, do_sample = True) #True instead of False

        #convert JSON -> Python dictionary
        data = json.loads(LLM_reply)

        #validate dictionary with Pydantic
        input_parameters = InputParameters.model_validate(data)

        return LLM_reply, input_parameters

class VerifyLLM(LLM):

    def verify_information(self, user_message: str, input_parameters: InputParameters, max_new_tokens: int = 1000) -> str:

        input_parameters_json = input_parameters.model_dump_json(indent=2)

        message = f"""ORIGINAL USER MESSAGE:
                        {user_message}

                        EXTRACTED PARAMETERS:
                        {input_parameters_json}

                        Verify whether the extracted parameters accurately represent the information
                        explicitly provided in the original user message."""

        LLM_reply = self.generate(message, max_new_tokens = max_new_tokens, do_sample=True) #true instead of false

        return LLM_reply

class UpdateLLM(LLM):

    def update_information(self, message_update: str, current_input_parameters: InputParameters, max_new_tokens: int = 500) -> tuple[str, InputParameters]:

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        message = f"""UPDATE INSTRUCTION:
                    {message_update}

                    CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Update the current parameters according to the update instruction.
                    Return the complete updated JSON object.
                    """

        updated_json = self.generate(message, max_new_tokens=max_new_tokens, do_sample=True) #true instead of false for llama

        data = json.loads(updated_json)

        updated_input_parameters = InputParameters.model_validate(data)

        return updated_json, updated_input_parameters

class FillLLM(LLM):

    def __init__(self, model_manager: ModelManager, model_preprompt: str, database_path: str) -> None:

        super().__init__(model_manager, model_preprompt)

        self.database_path = database_path

    def fill_missing_information(self, current_input_parameters: InputParameters, max_new_tokens: int = 10000) -> tuple[str, InputParameters]:#previously 500 

        database = MechanismDatabase(self.database_path)

        results = database.find_best_matches(
                        current_input_parameters,
                        max_results=10, #provides only max 10 results
                    )

        # Report from database retrieval
        # print(f"Number of matches: {results['n_matches']}")

        matched_results = results["matches"]

        print("Matched results:")
        for case in matched_results:
            print(f"ID:       {case.get('id')}")

        
        matched_cases = [database.case_to_prompt_format(case)
                                        for case in matched_results]

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        matched_cases_json = json.dumps(matched_cases, indent=2)

        current_data = current_input_parameters.model_dump()

        fixed_fields = [
            field for field, value in current_data.items()
            if value is not None
        ]

        fillable_fields = [
            field for field, value in current_data.items()
            if value is None
        ]


        message = f"""CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    FIELDS THAT MUST NOT BE CHANGED:
                    {json.dumps(fixed_fields)}

                    FIELDS THAT MAY BE FILLED:
                    {json.dumps(fillable_fields)}

                    MATCHING CASES:
                    {matched_cases_json}

                    Complete the missing parameters according to your instructions.
                    Keep the exact values from CURRENT PARAMETERS and fill in the missing values using the MATCHING CASES according to your instructions.
                    """

        filled_json = self.generate(message, max_new_tokens=max_new_tokens, do_sample=True, enable_thinking=True) #true instead of false with llama
        print(filled_json)
        data = json.loads(filled_json)

        filled_input_parameters = InputParameters.model_validate(data)

        return filled_json, filled_input_parameters, matched_results

class RouterLLM(LLM):

    def define_route(self, agent_message: str, user_message: str, current_input_parameters: InputParameters, list_of_possible_routes: list[str], max_new_tokens: int = 500):

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        message = f"""LAST AGENT MESSAGE:
                    {agent_message}
        
                    USER MESSAGE:
                    {user_message}

                    CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Choose from the list below and output the most appropriate route based on the last agent message, the user message and the current set of parameters.

                    POSSIBLE ROUTES:
                    {list_of_possible_routes}

                    Only choose from this list. You cannot choose another option.
                    """

        chosen_route = self.generate(message, max_new_tokens=max_new_tokens, do_sample=False, enable_thinking=False) #changed vs LLama, sample true instead of False

        if chosen_route not in list_of_possible_routes:
            raise ValueError(f"LLM selected invalid route '{chosen_route}'. Expected one of: {list_of_possible_routes}.")

        return chosen_route