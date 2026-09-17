
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

## Inputs


model = 'data/mechanisms/detailed/Burke-2012-H2-N2.yaml' #'gri30.yaml'
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
num_threads=20
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


    

data = { 'model': model,
             'targets': targets, 
             'retained-species': retained, 
             'method': method,
             'error': max_error_IDT,
             'sensitivity-analysis': sensitivity,
             'autoignition-conditions':  cond_list_IDT
             }   
    
    
    
inputs = pymars.parse_inputs(data)
model_file = inputs.model
psr_conditions = inputs.psr_conditions
flame_conditions = inputs.plflame_conditions
ignition_conditions = inputs.ignition_conditions
upper_threshold=inputs.upper_threshold
target_species=inputs.target_species
safe_species=inputs.safe_species
error_limit = inputs.error

def main():

    inputs = pymars.parse_inputs(data)

     
    pfa.run_pfa(model_file, ignition_conditions, psr_conditions, flame_conditions, 
                    error_limit, target_species, safe_species, threshold_upper=None, num_threads=num_threads)
        
       

    drg.run_drg(model_file, ignition_conditions, psr_conditions, flame_conditions, 
                    error_limit, target_species, safe_species, threshold_upper=None, num_threads=num_threads)
        

        
    drgep.run_drgep(model_file, ignition_conditions, psr_conditions, flame_conditions, 
                    error_limit, target_species, safe_species, threshold_upper=None, num_threads=num_threads)
        
       

if __name__ == "__main__":
    main()
    
    

    
 



    