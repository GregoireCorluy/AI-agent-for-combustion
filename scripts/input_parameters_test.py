from CombustionAgent.parameters import InputParameters, FuelComponent

schema = InputParameters.model_json_schema()

print(schema)

print()

params = InputParameters(
    fuel=[
        FuelComponent(species="NH3", fraction=0.8),
        FuelComponent(species="H2", fraction=0.2),
    ],

    application_regime=[
        "premixed",
        "flame speed",
    ],

    pressure_start=1.0,
    pressure_end=1.0,
    pressure_unit="bar",

    temperature_start=1000,
    temperature_end=500.0,
    temperature_unit="K",

    equivalence_ratio_start=0.5,
    equivalence_ratio_end=2.0,

    retained_species=["NH3", "H2", "O2", "N2"],
    target_species=["NO"],
)

input_parameters_json = params.model_dump_json(indent=2)

print(input_parameters_json)