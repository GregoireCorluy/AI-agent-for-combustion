
## Import


import cantera as ct
import numpy as np
import yaml
import itertools
import logging
import os
import pymars.pymars as pymars
import pymars.drgep as drgep
import pymars.pfa as pfa
import pymars.drg as drg


## Pymars log

logging.basicConfig(level=logging.INFO)


class AgentToolMechReduction():

    def __init__(self):
             
        ## Inputs

        path = 'data/mechanisms/detailed/'
        model = path + 'mech01-GRI3.0-1999.yaml'
        gas = ct.Solution('gri30.yaml')
        targets = ['CH4', 'H2']
        retained = ['O2', 'N2', 'CH4', 'H2']
        max_error_IDT = 15 #%
        autoignition_kind = 'constant volume'
        fuel = [ {'CH4': 0.99, 'H2':0.01},     
                {'CH4': 0.8, 'H2':0.2}]
        oxidizer = {'O2': 1.0, 'N2': 3.76}
        equivalence_ratio = [0.5, 1.0, 2.0]
        temperature_IDT = [700,  1100, 1500, 1800, 2000]
        pressure = [1, 20, 50]
        self.num_threads=1
        method='DRGEP'
        sensitivity=False


        # Create inputs dictionary

        cond_list_IDT = []

        for P, T, fx, phi in itertools.product(pressure, temperature_IDT, fuel, equivalence_ratio):

                    #logging.basicConfig(level=logging.INFO)

                
                    condition = {
                        'kind': 'constant volume',
                        'pressure': P,
                        'temperature': T,
                        'fuel': fx,
                        'oxidizer': oxidizer,
                        'equivalence-ratio': phi
                        }
                    cond_list_IDT.append(condition)


            

        self.data = { 'model': model,
                    'targets': targets, 
                    'retained-species': retained, 
                    'method': method,
                    'error': max_error_IDT,
                    'sensitivity-analysis': sensitivity,
                    'autoignition-conditions':  cond_list_IDT
                    }   
            
            
            
        inputs = pymars.parse_inputs(self.data)
        self.model_file = inputs.model
        self.psr_conditions = inputs.psr_conditions
        self.flame_conditions = inputs.plflame_conditions
        self.ignition_conditions = inputs.ignition_conditions
        self.upper_threshold=inputs.upper_threshold
        self.target_species=inputs.target_species
        self.safe_species=inputs.safe_species
        self.error_limit = inputs.error

    def run_dgrep(self):
    
        drgep.run_drgep(self.model_file, self.ignition_conditions, self.psr_conditions, self.flame_conditions, 
                        self.error_limit, self.target_species, self.safe_species, threshold_upper=None, num_threads=self.num_threads, path='data/temp/')