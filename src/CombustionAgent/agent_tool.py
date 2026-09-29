
## Import


import cantera as ct
import numpy as np
import yaml
import itertools
import logging
import os
import pickle
from datetime import date
import pymars.pymars as pymars
import pymars.drgep as drgep
import pymars.pfa as pfa
import pymars.drg as drg
from CombustionAgent.parameters import InputParameters
from pymars.drgep import retrieve_next_ID

## Pymars log

logging.basicConfig(level=logging.INFO)


class AgentToolMechReduction():

    def __init__(self):
             
        ## Inputs

        self.path = 'data/detailed_mechanisms/'
        self.max_error_IDT = [5, 10, 20] #%
        self.oxidizer = {'O2': 1.0, 'N2': 3.76}
        self.autoignition_kind = 'constant volume'
        self.num_threads=1
        self.method='DRGEP'
        self.sensitivity=False

        self.setup_directories()

    def setup_directories(self):
        """Create required directories and clean temporary files."""

        os.makedirs("outputs/reduced_mechanisms/meta", exist_ok=True)
        os.makedirs("outputs/reduced_mechanisms/metrics", exist_ok=True)
        os.makedirs("temp", exist_ok=True)

        # Clean temporary files from previous runs
        for item in os.listdir("temp"):
            os.remove(os.path.join("temp", item))

    def get_data_IDT(self, input_parameters: InputParameters):
                # Create inputs dictionary

        mechanism = input_parameters.mechanism

        #fuel = ...

        temperature_start = input_parameters.temperature_start
        temperature_end = input_parameters.temperature_end

        pressure_start = input_parameters.pressure_start
        pressure_end = input_parameters.pressure_end

        equivalence_ratio_start = input_parameters.equivalence_ratio_start
        equivalence_ratio_end = input_parameters.equivalence_ratio_end

        model = self.path + mechanism + '.yaml'
        targets = input_parameters.target_species
        retained = input_parameters.retained_species

        #Here for the moment can only handle one mixture
        fuel = [{component.species: component.fraction for component in input_parameters.fuel}] #Need to modify later to explore different fuel compositions
        temperature_IDT = (np.array([temperature_start]) if temperature_start == temperature_end else np.linspace(temperature_start, temperature_end, 3))
        pressure = (np.array([pressure_start]) if pressure_start == pressure_end else np.linspace(pressure_start, pressure_end, 3))
        equivalence_ratio = (np.array([equivalence_ratio_start]) if equivalence_ratio_start == equivalence_ratio_end else np.linspace(equivalence_ratio_start, equivalence_ratio_end, 3))

        print("\n" + "=" * 60)
        print("DRGEP REDUCTION CONDITIONS")
        print("=" * 60)

        print(f"Mechanism          : {mechanism}.yaml")
        print(f"Fuel               : {fuel}")
        print(f"Retained species   : {retained}")
        print(f"Target species     : {targets}")
        print(f"Temperature [K]    : {temperature_IDT}")
        print(f"Pressure [bar]     : {pressure}")
        print(f"Equivalence ratio  : {equivalence_ratio}")
        print(f"Maximum error      : {self.max_error_IDT}%")

        print("=" * 60 + "\n")

        cond_list_IDT = []

        for P, T, fx, phi in itertools.product(pressure, temperature_IDT, fuel, equivalence_ratio):
             
                    condition = {
                        'kind': self.autoignition_kind,
                        'pressure': P,
                        'temperature': T,
                        'fuel': fx,
                        'oxidizer': self.oxidizer,
                        'equivalence-ratio': phi
                        }
                    cond_list_IDT.append(condition)
            
        data_list = []

        for error_IDT in self.max_error_IDT:
            data = { 'model': model,
                        'targets': targets, 
                        'retained-species': retained, 
                        'method': self.method,
                        'error': error_IDT,
                        'sensitivity-analysis': self.sensitivity,
                        'autoignition-conditions':  cond_list_IDT
                        }
            data_list.append(data)

        return data_list
            
    def run_drgep(self, input_parameters: InputParameters) -> list[str]:

        self.data_list = self.get_data_IDT(input_parameters)

        path = "outputs/reduced_mechanisms/meta/"
        today = date.today().strftime("%Y-%m-%d")
        ID = retrieve_next_ID()

        list_mechanisms = []

        for data in self.data_list:
            inputs = pymars.parse_inputs(data)
            self.model_file = inputs.model
            self.psr_conditions = inputs.psr_conditions
            self.flame_conditions = inputs.plflame_conditions
            self.ignition_conditions = inputs.ignition_conditions
            self.upper_threshold=inputs.upper_threshold
            self.target_species=inputs.target_species
            self.safe_species=inputs.safe_species
            self.error_limit = inputs.error

            data_pickle = {
                "model_file": self.model_file,
                "psr_conditions": self.psr_conditions,
                "flame_conditions": self.flame_conditions,
                "ignition_conditions": self.ignition_conditions,
                "upper_threshold": self.upper_threshold,
                "target_species": self.target_species,
                "safe_species": self.safe_species,
                "error_limit": self.error_limit,
            }

            name_mechanism = today + f"-{ID}-" + input_parameters.mechanism + f"-error{self.error_limit}"
            list_mechanisms.append(name_mechanism)

            with open(path + name_mechanism + "-meta.pkl", "wb") as f:
                pickle.dump(data_pickle, f)
        
            drgep.run_drgep(self.model_file, self.ignition_conditions, self.psr_conditions, self.flame_conditions, 
                            self.error_limit, self.target_species, self.safe_species, threshold_upper=None, num_threads=self.num_threads, path='temp/', id = ID)

        return list_mechanisms