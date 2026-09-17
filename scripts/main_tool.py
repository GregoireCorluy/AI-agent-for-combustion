from CombustionAgent.agent_tool import AgentToolMechReduction

import pymars.pymars
import numexpr
import sys

def main():

    agent_tool_mechanism_reduction = AgentToolMechReduction()
    agent_tool_mechanism_reduction.run_dgrep()


if __name__ == "__main__":
    main()