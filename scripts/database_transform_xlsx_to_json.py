import pandas as pd
import json


INPUT_FILE = "data/database/2026-09-14-database-mechanisms-H2-NH3-preliminary.xlsx"
OUTPUT_FILE = "data/database/2026-09-14-database-mechanisms-H2-NH3-preliminary.json"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def split_list(value):
    """Convert a comma-separated Excel cell into a Python list."""
    if pd.isna(value):
        return []

    return [item.strip() for item in str(value).split(",")]


def clean_value(value):
    """Convert pandas NaN to None."""
    if pd.isna(value):
        return None
    return value


# --------------------------------------------------
# Read Excel
# --------------------------------------------------

df = pd.read_excel(INPUT_FILE, sheet_name = "v2.1 Modified")


# --------------------------------------------------
# Convert rows
# --------------------------------------------------

cases = []

for _, row in df.iterrows():

    case = {
        "id": clean_value(row["ID"]),

        "fuel": split_list(row["Fuel"]),
        "oxidizer": split_list(row["Oxidizer"]),

        "application_regime": split_list(
            row["Application/Regime"]
        ),

        "temperature": {
            "min": clean_value(row["T0_min"]),
            "max": clean_value(row["T0_max"]),
            "unit": "K"
        },

        "equivalence_ratio": {
            "min": clean_value(row["phi_min"]),
            "max": clean_value(row["phi_max"])
        },

        "pressure": {
            "min": clean_value(row["P_min"]),
            "max": clean_value(row["P_max"]),
            "unit": "bar"
        },

        "mechanism": clean_value(row["Mechanism"]),

        "retained_species": split_list(
            row["Retained species"]
        ),

        "target_species": split_list(
            row["Target species"]
        )
    }

    cases.append(case)


# --------------------------------------------------
# Create JSON structure
# --------------------------------------------------

database = {
    "cases": cases
}


# --------------------------------------------------
# Write JSON
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        database,
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"Database written to: {OUTPUT_FILE}")
print(f"Number of cases: {len(cases)}")