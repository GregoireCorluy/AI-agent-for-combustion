from CombustionAgent.parameters import InputParameters, FuelComponent

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


if all(value is not None  for field, value in input_parameters.model_dump().items() if field != "application_regime"):
    print("If-loop succeedded")
else:
    print("Not in statement")