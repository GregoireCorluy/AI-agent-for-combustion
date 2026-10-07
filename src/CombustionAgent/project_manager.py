import questionary
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from .parameters import InputParameters, CriteriaParameters


class ProjectManager:

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)

        self.selected_project_path: Path | None = None

    def select_project(self):

        # clean and remove empty projects
        self.clean_projects()

        projects = self.get_projects()
        new_project = False

        choices = [
            questionary.Choice(
                title=project["name"],
                value=project["id"]
            )
            for project in projects
        ]

        choices.insert(
            0,
            questionary.Choice(
                title="New project",
                value="new_project"
            )
        )

        selected_project_id = questionary.select(
            "Select a project:",
            choices=choices
        ).ask()

        if selected_project_id == "new_project":

            project_name = questionary.text(
                "Enter a name for the project:"
            ).ask()

            selected_project_id = self.create_new_project(project_name)

            new_project = True

        self.selected_project_path = self.project_path / selected_project_id

        return selected_project_id, new_project

    def get_projects(self):

        projects = []

        for project_dir in self.project_path.iterdir():

            if not project_dir.is_dir():
                continue

            project_file = project_dir / "project.json"

            if not project_file.is_file():
                continue

            with project_file.open("r", encoding="utf-8") as f:
                project_data = json.load(f)

            projects.append(project_data)

        return projects

    def create_new_project(self, name_project):

        # Create project ID: YYYYMMDD_001
        date = datetime.now().strftime("%Y%m%d")

        existing_ids = [
            path.name
            for path in self.project_path.iterdir()
            if path.is_dir() and path.name.startswith(date + "_")
        ]

        numbers = []
        for project_id in existing_ids:
            try:
                numbers.append(int(project_id.split("_")[1]))
            except (IndexError, ValueError):
                pass

        next_number = max(numbers, default=0) + 1
        project_id = f"{date}_{next_number:03d}"

        # Create project directory
        project_dir = self.project_path/ project_id
        project_dir.mkdir(parents=True, exist_ok=False)

        # Create project metadata file
        project_file = project_dir / "project.json"

        project_data = {
            "id": project_id,
            "name": name_project
        }

        with project_file.open("w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=4)

        return project_id

    def clean_projects(self):

        for project_dir in self.project_path.iterdir():

            if not project_dir.is_dir():
                continue

            project_file = project_dir / "project.json"

            if not project_file.is_file():
                continue

            iteration_files = list(project_dir.glob("iteration_*.json"))

            if not iteration_files:
                shutil.rmtree(project_dir)


    def save_data(self, messages_history: list[str],
                  behind_the_scene_history: list[str],
                  input_parameters: InputParameters,
                  criteria_parameters: CriteriaParameters,
                  mechanisms_metrics_json: str,
                  selection_mechanism: str) -> None:



        # Directory containing the project files
        project_dir = Path(self.selected_project_path)

        # Find existing iteration files
        iteration_files = list(project_dir.glob("iteration_*.json"))

        # Determine next iteration number
        if not iteration_files:
            iteration_number = 1
        else:
            iteration_numbers = []

            for file in iteration_files:
                match = re.fullmatch(r"iteration_(\d+)\.json", file.name)

                if match:
                    iteration_numbers.append(int(match.group(1)))

            iteration_number = max(iteration_numbers, default=0) + 1

        # Create iteration filename
        iteration_filename = f"iteration_{iteration_number:03d}.json"
        iteration_path = project_dir / iteration_filename

        # Convert Pydantic models to dictionaries
        input_parameters_dict = input_parameters.model_dump()
        criteria_parameters_dict = criteria_parameters.model_dump()

        # Convert mechanisms metrics from JSON string to Python object
        mechanisms_metrics = json.loads(mechanisms_metrics_json)

        # Create summary for the future agent
        # Save what has been modified vs previous iteration?
        summary = {
            "input_parameters": input_parameters_dict,
            "criteria_parameters": criteria_parameters_dict,
            "mechanisms": mechanisms_metrics,
            "selection": selection_mechanism
        }

        # Create complete iteration data
        iteration_data = {
            "iteration": iteration_number,
            "discussion": messages_history,
            "behind_the_scene": behind_the_scene_history,
            "input_parameters": input_parameters_dict,
            "criteria_parameters": criteria_parameters_dict,
            "mechanisms": mechanisms_metrics,
            "selection": selection_mechanism,
            "summary": summary
        }

        # Save JSON
        with open(iteration_path, "w", encoding="utf-8") as f:
            json.dump(
                iteration_data,
                f,
                indent=4,
                ensure_ascii=False
            )

        return None

    def get_previous_summaries(self) -> list[dict]:

        project_dir = Path(self.selected_project_path)

        iteration_files = sorted(
            project_dir.glob("iteration_*.json"),
            key=lambda path: int(
                re.fullmatch(r"iteration_(\d+)\.json", path.name).group(1)
            )
        )

        summaries = []

        for iteration_file in iteration_files:

            with iteration_file.open("r", encoding="utf-8") as f:
                iteration_data = json.load(f)

            if "summary" in iteration_data:
                summaries.append(iteration_data["summary"])

        return summaries