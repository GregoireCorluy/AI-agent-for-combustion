from CombustionAgent.parameters import InputParameters, FuelComponent
from CombustionAgent.graph import AgentGraph


# ============================================================
# Database
# ============================================================

database_path = "data/database/2026-09-14-database-mechanisms-H2-NH3-preliminary.json"

agent_graph = AgentGraph(
    agent=None,
    database_path=database_path,
)


# ============================================================
# User input
# ============================================================

params = InputParameters(
    mechanism = "paris",
    fuel=[
        FuelComponent(species="NH3", fraction=0.8),
        #FuelComponent(species="H2", fraction=0.2),
    ],

    # application_regime=[
    #     "premixed",
    #     "flame speed",
    # ],

    # pressure_start=1.0,
    # pressure_end=1.0,
    # pressure_unit="bar",

    temperature_start=1000,
    # temperature_end=500.0,
    temperature_unit="K",

    # equivalence_ratio_start=0.5,
    # equivalence_ratio_end=2.0,

    # retained_species=["NH3", "H2", "O2", "N2"],
    # target_species=["NO"],
)


# ============================================================
# Find matching mechanism name
# ============================================================

matched_name, messages = agent_graph.convert_mechanism_name(
    params)


# ============================================================
# Print results
# ============================================================

print(matched_name)
print(messages)