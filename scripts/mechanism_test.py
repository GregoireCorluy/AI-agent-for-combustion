from CombustionAgent.agent_tool import AgentToolMechReduction
from CombustionAgent.parameters import InputParameters, FuelComponent

params = InputParameters(
    mechanism = "Glarborg-2018-H2-only-HNO",

    # "Konnov-2025-NH3" (add AR as retained species?) #"Glarborg-2018-H2-only-HNO", "Glarborg-2024-NH3", "Li-Konnov-2019-NH3", "Otomo-2018-NH3", "CRECK-2012-H2-H-v1212", "Burke-2012-H2-N2"

    fuel=[
        FuelComponent(species="H2", fraction=1.0),
    ],

    application_regime=[
        "premixed",
        "flame speed",
    ],

    pressure_start=1.0,
    pressure_end=2.0,
    pressure_unit="bar",

    temperature_start=1000,
    temperature_end=1500,
    temperature_unit="K",

    equivalence_ratio_start=0.5,
    equivalence_ratio_end=2.0,

    retained_species=["H2", "O2", "N2", "AR"],
    target_species=["H2"],)

agent_tool_mechanism_reduction = AgentToolMechReduction()

agent_tool_mechanism_reduction.run_dgrep(params)