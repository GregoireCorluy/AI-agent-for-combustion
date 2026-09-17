from pydantic import BaseModel, Field

class FuelComponent(BaseModel):
    species: str = Field(
        description="Fuel species in chemical notation, e.g., CH4 for methane, NH3 for ammonia or H2 for hydrogen."
    )
    fraction: float = Field(
        description="Fuel fraction of this species. Coefficient needs to be between 0 and 1. Fractions must sum to 1."
    )

class InputParameters(BaseModel):

    mechanism: str | None = Field(default = None, description = "Chemical mechanism") #give a list of possible mechanisms? Database for the mechanisms?
    application_regime: list[str] | None = Field(default = None, description = "List of keywords describing the application or regime of the combustion case/simulation.")
    fuel: list[FuelComponent] | None = Field(default = None, description = "Fuel composition. Each component explicitly associates a fuel species with its fraction.")

    pressure_start: float | None = Field(default = None, description = "Lowerbound of the pressure at which the simulation is performed.")
    pressure_end: float | None = Field(default = None, description = "Upperbound of the pressure at which the simulation is performed.")
    pressure_unit: str | None = Field(default = None, description = "Unit of the pressure provided by the user. E.g., 'P' for Pascal. 'atm' for standard atmosphere, 'bar'.")

    temperature_start: float | None = Field(default = None, description = "Lowerbound of the temperature at which the simulation should be performed, keeping the number that is provided by the user. Do not convert to another unit.")
    temperature_end: float | None = Field(default = None, description = "Upperbound of the temperature at which the simulation should be performed, keeping the number that is provided by the user. Do not convert to another unit.")
    temperature_unit: str | None = Field(default = None, description = "Unit of the temperature provided by the user. E.g., 'K' for Kelvin. 'C' for Celsius. 'F' for Fahrenheit.")

    equivalence_ratio_start: float | None = Field(default = None, description = "Lowerbound of the equivalence ratio range")
    equivalence_ratio_end: float | None = Field(default = None, description = "Upperbound of the equivalence ratio range")

    retained_species: list[str] | None = Field(default = None, description = "Retained species. Species that absolutely need to be retained in the mechanism")
    target_species: list[str] | None = Field(default = None, description = "Target species. Species that the mechanism need to model and capture correctly.")

    # # Function to validate that the composition sums up to one
    # @field_validator("fuel")
    # @classmethod
    # def validate_fuel_composition(cls, value):
    #     if value is None:
    #         return value

    #     total = sum(component.fraction for component in value)

    #     if abs(total - 1.0) > 1e-6:
    #         raise ValueError(
    #             f"Fuel fractions must sum to 1. Got {total}."
    #         )

    #     return value