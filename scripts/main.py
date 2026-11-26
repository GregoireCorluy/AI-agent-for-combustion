"""Command-line entry point for the tool-using CombustionAgent."""
import argparse

from CombustionAgent.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Combustion mechanism selection and reduction assistant")
    parser.add_argument("--orchestrator-model", default=None,
                        help="Override COMBUSTION_ORCHESTRATOR_MODEL (default: Qwen/Qwen3.8-27B)")
    parser.add_argument("--database", default="data/database/2026-09-14-database-mechanisms-H2-NH3-preliminary.json")
    parser.add_argument("--projects", default="projects/")
    args = parser.parse_args()
    agent = Agent(model_name=args.orchestrator_model,
                  database_path=args.database,
                  project_path=args.projects)
    agent.main()


if __name__ == "__main__":
    main()
