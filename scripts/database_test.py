from CombustionAgent.parameters import InputParameters, FuelComponent
from CombustionAgent.database import MechanismDatabase


# ============================================================
# Database
# ============================================================

database_path = "data/database/2026-09-14-database-mechanisms-H2-NH3-preliminary.json"

database = MechanismDatabase(database_path)


# ============================================================
# User input
# ============================================================

params = InputParameters(
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
# Find matching cases
# ============================================================

results = database.find_best_matches(
    params,
    max_results=10, #provides only max 10 results
)


# ============================================================
# Print results
# ============================================================

print("\n==============================")
print("INPUT PARAMETERS")
print("==============================")

print(params)


print("\n==============================")
print("MATCHING RESULTS")
print("==============================")

print(f"Number of matches: {results['n_matches']}")

for case in results["matches"]:
    print("\n------------------------------")
    print(f"ID:       {case.get('id')}")
    print(f"Fuel:     {case.get('fuel')}")
    print(f"Application: {case.get('application_regime')}")
    print(f"Mechanism: {case.get('mechanism')}")


print("\n==============================")
print("MATCHING STEPS")
print("==============================")

for step in results["matching_steps"]:
    print(
        f"{step['criterion']:25s} | "
        f"applied={step['applied']} | "
        f"remaining={step['remaining']}"
    )