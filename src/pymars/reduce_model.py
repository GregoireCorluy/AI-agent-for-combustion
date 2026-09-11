"""Module for general model reduction class and function."""
import os
import re
from typing import NamedTuple

import cantera as ct

class ReducedModel(NamedTuple):
    """Represents reduced model and associated metadata
    """
    model: ct.Solution
    filename: str = ''
    error_idt: float = 0.0
    error_psr: float = 0.0
    error_plflame: float = 0.0
    limbo_species: list = []
    

def trim(initial_model_file, exclusion_list, new_model_file, phase_name=''):
    """Function to eliminate species and corresponding reactions from model

    Parameters
    ----------
    initial_model_file : str
        Filename for initial model to be reduced
    exclusion_list : list of str
        List of species names that will be removed
    new_model_file : str
        Name of new reduced model file
    phase_name : str, optional
        Optional name for phase to load from CTI file (e.g., 'gas'). 

    Returns
    -------
    new_solution : ct.Solution
        Model with species and associated reactions eliminated

    """
    solution = ct.Solution(initial_model_file, phase_name)

    # Remove species if in list to be removed
    final_species = [sp for sp in solution.species() if sp.name not in exclusion_list]
    final_species_names = [sp.name for sp in final_species]

    # Remove reactions that use eliminated species
    final_reactions = []

    for reaction in solution.reactions():
    
        # Collect reactant + product species
        eq = reaction.equation
        eq = eq.replace("(+M)", "")
        eq = eq.replace("(M)", "")
        eq = eq.replace("<=>", " + ")
        eq = eq.replace("=>", " + ")
        eq = eq.replace("=", " + ")
        eq = eq.replace("(", "")
        eq = eq.replace(")", "")

       
         

        reaction_species = [    
        token.strip()
        for token in eq.split("+")
        if token.strip() in solution.species_names
        ]
        
        reaction_species_1 = list(reaction.reactants.keys() | reaction.products.keys())

        reaction_species = reaction_species  + [spec for spec in reaction_species_1 if spec not in reaction_species]

        if '(+HE)' in eq and 'HE' not in reaction_species:
             reaction_species.append('HE')
             
    
        # Skip reaction if it contains a removed species
        if not set(reaction_species).issubset(final_species_names):
            continue
    
        # --- Proper third-body cleaning (modern Cantera) ---
        if hasattr(reaction, "third_body") and reaction.third_body is not None:
            reaction.third_body.efficiencies = {
                sp: val
                for sp, val in reaction.third_body.efficiencies.items()
                if sp in final_species_names
            }
    
        final_reactions.append(reaction)

    # Create new solution based on remaining species and reactions
    new_solution = ct.Solution(
        species=final_species, reactions=final_reactions,
        thermo='IdealGas', kinetics='GasKinetics'
        )
    new_solution.TP = solution.TP
    if phase_name:
        new_solution.name = phase_name
    else:
        new_solution.name = os.path.splitext(new_model_file)[0]

    return new_solution

