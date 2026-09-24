import json
from pathlib import Path
from typing import Any

from .parameters import InputParameters, FuelComponent


class MechanismDatabase:

    def __init__(self, database_path: str | Path):

        self.database_path = Path(database_path)

        with open(self.database_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.cases = data["cases"]

    # ============================================================
    # Public method
    # ============================================================

    def get_unique_mechanisms(self) -> list[str]:
        """Return the unique mechanism names stored in the database."""
        
        return sorted({
            case["mechanism"]
            for case in self.cases
            if case.get("mechanism")
        })

    def get_unique_application_regime(self) -> list[str]:
        """Return the unique application/regime names stored in the database."""
        
        return sorted({
            regime
            for case in self.cases
            for regime in case.get("application_regime", [])
        })
    
    def find_best_matches(
        self,
        params: InputParameters,
        max_results: int = 3,
    ) -> dict[str, Any]:

        candidates = self.cases.copy()

        matching_steps = []

        # --------------------------------------------------------
        # 1. Fuel
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._fuel_match(case, params.fuel),
        )

        matching_steps.append({
            "criterion": "fuel",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 2. Application / regime
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._application_regime_match(
                case,
                params.application_regime,
            ),
        )

        matching_steps.append({
            "criterion": "application_regime",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 3. Pressure
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._pressure_match(
                case,
                params,
            ),
        )

        matching_steps.append({
            "criterion": "pressure",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 4. Temperature
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._temperature_match(
                case,
                params,
            ),
        )

        matching_steps.append({
            "criterion": "temperature",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 5. Equivalence ratio
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._phi_match(
                case,
                params,
            ),
        )

        matching_steps.append({
            "criterion": "equivalence_ratio",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 6. Retained species
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._species_match(
                case.get("retained_species", []),
                params.retained_species,
            ),
        )

        matching_steps.append({
            "criterion": "retained_species",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 7. Target species
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._species_match(
                case.get("target_species", []),
                params.target_species,
            ),
        )

        matching_steps.append({
            "criterion": "target_species",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # 8. Mechanism
        # --------------------------------------------------------

        candidates, applied = self._apply_filter(
            candidates,
            lambda case: self._mechanism_match(
                case,
                params.mechanism,
            ),
        )

        matching_steps.append({
            "criterion": "mechanism",
            "applied": applied,
            "remaining": len(candidates),
        })

        # --------------------------------------------------------
        # Return top candidates
        # --------------------------------------------------------

        return {
            "matches": candidates[:max_results],
            "n_matches": len(candidates),
            "matching_steps": matching_steps,
        }

    # ============================================================
    # Generic filtering logic
    # ============================================================

    @staticmethod
    def _apply_filter(
        candidates: list[dict],
        filter_function,
    ) -> tuple[list[dict], bool]:

        matches = [
            case
            for case in candidates
            if filter_function(case)
        ]

        # IMPORTANT:
        # Only apply the filter if at least one candidate matches.
        #
        # If zero candidates match, keep the original candidate set.

        if matches:
            return matches, True

        return candidates, False

    # ============================================================
    # Fuel
    # ============================================================

    @staticmethod
    def _fuel_match(
        case: dict,
        user_fuel: list[FuelComponent] | None,
    ) -> bool:

        if not user_fuel:
            return False

        # Fuel names stored in the database
        database_fuels = {
            fuel.strip().upper()
            for fuel in case.get("fuel", [])
        }

        # Extract only the species names from the user's fuel composition
        user_fuels = {
            component.species.strip().upper()
            for component in user_fuel
        }

        # Match only the fuel species, ignoring composition
        return database_fuels == user_fuels

    # ============================================================
    # Application / regime
    # ============================================================

    @staticmethod
    def _application_regime_match(
        case: dict,
        user_keywords: list[str] | None,
    ) -> bool:

        if not user_keywords:
            return False

        database_keywords = {
            keyword.strip().lower()
            for keyword in case.get("application_regime", [])
        }

        user_keywords = {
            keyword.strip().lower()
            for keyword in user_keywords
        }

        # Match if at least one user keyword is present
        # in the database keywords.
        return bool(database_keywords.intersection(user_keywords))

    # ============================================================
    # Pressure
    # ============================================================

    @staticmethod
    def _pressure_match(
        case: dict,
        params: InputParameters,
    ) -> bool:

        if (
            params.pressure_start is None
            and params.pressure_end is None
        ):
            return False

        # Currently database pressure is in bar.
        if params.pressure_unit is None:
            return False

        if params.pressure_unit.lower() != "bar":
            return False

        user_min = (
            params.pressure_start
            if params.pressure_start is not None
            else params.pressure_end
        )

        user_max = (
            params.pressure_end
            if params.pressure_end is not None
            else params.pressure_start
        )

        db_min = case["pressure"]["min"]
        db_max = case["pressure"]["max"]

        return _ranges_overlap(
            user_min,
            user_max,
            db_min,
            db_max,
        )

    # ============================================================
    # Temperature
    # ============================================================

    @staticmethod
    def _temperature_match(
        case: dict,
        params: InputParameters,
    ) -> bool:

        if (
            params.temperature_start is None
            and params.temperature_end is None
        ):
            return False

        # Database is currently stored in Kelvin.
        if params.temperature_unit is None:
            return False

        if params.temperature_unit.upper() != "K":
            return False

        user_min = (
            params.temperature_start
            if params.temperature_start is not None
            else params.temperature_end
        )

        user_max = (
            params.temperature_end
            if params.temperature_end is not None
            else params.temperature_start
        )

        db_min = case["temperature"]["min"]
        db_max = case["temperature"]["max"]

        return _ranges_overlap(
            user_min,
            user_max,
            db_min,
            db_max,
        )

    # ============================================================
    # Equivalence ratio
    # ============================================================

    @staticmethod
    def _phi_match(
        case: dict,
        params: InputParameters,
    ) -> bool:

        if (
            params.equivalence_ratio_start is None
            and params.equivalence_ratio_end is None
        ):
            return False

        user_min = (
            params.equivalence_ratio_start
            if params.equivalence_ratio_start is not None
            else params.equivalence_ratio_end
        )

        user_max = (
            params.equivalence_ratio_end
            if params.equivalence_ratio_end is not None
            else params.equivalence_ratio_start
        )

        db_min = case["equivalence_ratio"]["min"]
        db_max = case["equivalence_ratio"]["max"]

        return _ranges_overlap(
            user_min,
            user_max,
            db_min,
            db_max,
        )

    # ============================================================
    # Species
    # ============================================================

    @staticmethod
    def _species_match(
        database_species: list[str],
        user_species: list[str] | None,
    ) -> bool:

        if not user_species:
            return False

        database_species = {
            species.strip().upper()
            for species in database_species
        }

        user_species = {
            species.strip().upper()
            for species in user_species
        }

        # All species requested by the user must be present
        # in the database case.
        #return user_species.issubset(database_species)

        # At least one species correspond
        return bool(user_species & database_species)

    # ============================================================
    # Mechanism
    # ============================================================

    @staticmethod
    def _mechanism_match(
        case: dict,
        user_mechanism: str | None,
    ) -> bool:

        if user_mechanism is None:
            return False

        return (
            case.get("mechanism", "").lower().strip()
            == user_mechanism.lower().strip()
        )

    @staticmethod
    def case_to_prompt_format(case: dict) -> dict:

        database_fuel = case.get("fuel") or []

        if database_fuel:
            fuel_fraction = 1.0 / len(database_fuel)

            formatted_fuel = [
                {
                    "species": species,
                    "fraction": fuel_fraction,
                }
                for species in database_fuel
            ]
        else:
            formatted_fuel = None

        return {
            "mechanism": case.get("mechanism"),
            "application_regime": case.get("application_regime"),
            "fuel": formatted_fuel,

            "pressure_start": case.get("pressure", {}).get("min"),
            "pressure_end": case.get("pressure", {}).get("max"),
            "pressure_unit": case.get("pressure", {}).get("unit"),

            "temperature_start": case.get("temperature", {}).get("min"),
            "temperature_end": case.get("temperature", {}).get("max"),
            "temperature_unit": case.get("temperature", {}).get("unit"),

            "equivalence_ratio_start": case.get("equivalence_ratio", {}).get("min"),
            "equivalence_ratio_end": case.get("equivalence_ratio", {}).get("max"),

            "retained_species": case.get("retained_species"),
            "target_species": case.get("target_species"),
        }


# ================================================================
# Utility
# ================================================================

def _ranges_overlap(
    user_min: float,
    user_max: float,
    db_min: float,
    db_max: float,
) -> bool:

    return (
        user_min <= db_max
        and user_max >= db_min
    )