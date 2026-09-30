import questionary
import json
from pathlib import Path
from datetime import datetime


class ProjectManager:

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)

    def select_project(self):

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