"""Module containing sensitivity analysis reduction stage. """
import logging

import numpy as np
import cantera as ct

#from . import soln2cti
#from .sampling import sample_metrics, calculate_error, read_metrics
#from .reduce_model import trim, ReducedModel

from .sampling import sample_metrics, calculate_error, calculate_error_psr, calculate_error_plflame, read_metrics
from .reduce_model import trim, ReducedModel


# Taken from http://stackoverflow.com/a/22726782/1569494
try:
    from tempfile import TemporaryDirectory
except ImportError:
    from contextlib import contextmanager
    import shutil
    import tempfile
    import errno

    @contextmanager
    def TemporaryDirectory():
        name = tempfile.mkdtemp()
        try:
            yield name
        finally:
            try:
                shutil.rmtree(name)
            except OSError as e:
                # Reraise unless ENOENT: No such file or directory
                # (ok if directory has already been deleted)
                if e.errno != errno.ENOENT:
                    raise


def evaluate_species_errors(starting_model, ignition_conditions, psr_conditions, plflame_conditions, metrics_idt, 
                            metrics_psr, metrics_plflame,  species_limbo, 
                            phase_name='', num_threads=1
                            ):

    #print('starting model evaluate_species_errors:    ', starting_model.filename)
    """Calculate error induced by removal of each limbo species

    Parameters
    ----------
    starting_model : ReducedModel
        Container with model and file information
    ignition_conditions : list of InputIgnition
        List of autoignition initial conditions.
    metrics : numpy.ndarray
        Calculated metrics for starting model, used for evaluating error
    species_limbo : list of str
        List of species to consider removal
    phase_name : str, optional
        Optional name for phase to load from CTI file (e.g., 'gas'). 
    num_threads : int, optional
        Number of CPU threads to use for performing simulations in parallel.
        Optional; default = 1, in which the multiprocessing module is not used.
        If 0, then use the available number of cores minus one. Otherwise,
        use the specified number of threads.
    
    Returns
    -------
    species_errors : numpy.ndarray
        Maximum errors induced by removal of each limbo species

    """
    species_errors_idt = np.zeros(len(species_limbo))
    species_errors_psr = np.zeros(len(species_limbo))
    species_errors_plflame = np.zeros(len(species_limbo))


    with TemporaryDirectory() as temp_dir:
        for idx, species in enumerate(species_limbo):
            print('eliminating: ' ,species)
            test_model = trim(
                starting_model.filename, [species], f'reduced_model_{species}.yaml', 
                phase_name=phase_name
                )
            #test_model_file = soln2cti.write(
            #    test_model, f'reduced_model_{species}.yaml', path=temp_dir
            #    )
            test_model.write_yaml(f'reduced_model_{species}.yaml')
            test_model_file = f'reduced_model_{species}.yaml'

            #print('test_model_file evaluate_species_errors:    ', test_model_file)

            reduced_model_metrics_idt, reduced_model_metrics_psr, reduced_model_metrics_plflame = sample_metrics(
                test_model_file, ignition_conditions, psr_conditions, plflame_conditions, phase_name=phase_name, 
                num_threads=num_threads
                )
            species_errors_idt[idx] = 0
            species_errors_psr[idx] = 0
            species_errors_plflame[idx] = 0


            if ignition_conditions:
                species_errors_idt[idx] = calculate_error(metrics_idt, reduced_model_metrics_idt)

            if psr_conditions:
                species_errors_psr[idx] = calculate_error_psr(metrics_psr, reduced_model_metrics_psr)

            if plflame_conditions:
                species_errors_plflame[idx] = calculate_error_plflame(metrics_plflame, reduced_model_metrics_plflame)

    

    
    return species_errors_idt, species_errors_psr, species_errors_plflame


def run_sa(model_file, starting_error_idt, starting_error_psr, starting_error_plflame, ignition_conditions, 
           psr_conditions, plflame_conditions,
           error_limit, species_safe, phase_name='', algorithm_type='greedy', species_limbo=[],
           num_threads=1, path=''
           ):

    #print('model file sa:  ', model_file)
    """Runs a sensitivity analysis to remove species on a given model.
    
    Parameters
    ----------
    model_file : str
        Model being analyzed
    starting_error : float
        Error percentage between the reduced and original models
    ignition_conditions : list of InputIgnition
        List of autoignition initial conditions.
    psr_conditions : list of InputPSR, optional
        List of PSR simulation conditions.
    flame_conditions : list of InputLaminarFlame, optional
        List of laminar flame simulation conditions.
    error_limit : float
        Maximum allowable error level for reduced model
    species_safe : list of str
        List of species names to always be retained
    phase_name : str, optional
        Optional name for phase to load from CTI file (e.g., 'gas'). 
    algorithm_type : {'initial', 'greedy'}
        Type of sensitivity analysis: initial (order based on initial error), or 
        greedy (all species error re-evaluated after each removal)
    species_limbo : list of str, optional
        List of species to consider; if empty, consider all not in ``species_safe``
    num_threads : int, optional
        Number of CPU threads to use for performing simulations in parallel.
        Optional; default = 1, in which the multiprocessing module is not used.
        If 0, then use the available number of cores minus one. Otherwise,
        use the specified number of threads.
    path : str, optional
        Optional path for writing files
    
    Returns
    -------
    ReducedModel
        Return reduced model and associated metadata

    """
    current_model = ReducedModel(
        model=ct.Solution(model_file, phase_name), error_idt=starting_error_idt, error_psr=starting_error_psr, 
        error_plflame=starting_error_plflame, filename=model_file
        )


    #print('current model file sa:  ', current_model.filename)


    
    logging.info(f'Beginning sensitivity analysis stage, using {algorithm_type} approach.')

    # The metrics for the starting model need to be determined or read
    initial_metrics_idt, initial_metrics_psr, initial_metrics_plflame = sample_metrics(
        model_file, ignition_conditions, psr_conditions, plflame_conditions, reuse_saved=True, phase_name=phase_name,
        num_threads=num_threads, path=path
        )

    if not species_limbo:
        species_limbo = [
            sp for sp in current_model.model.species_names if sp not in species_safe
            ]
        #print('species_limbo inside:         ', species_limbo)
        #print('species_safe inside:         ', species_safe)


    logging.info(53 * '-')
    logging.info('Number of species |  Species removed  | Max error idt (%) | Max error psr (%) | Max error lbv (%)')

    #print('current:     ',  current_model.filename, ct.Solution(current_model.filename).n_species, ct.Solution(current_model.filename).n_reactions )

    # Need to first evaluate all induced errors of species; for the ``initial`` method,
    # this will be the only evaluation.
    species_errors_idt, species_errors_psr , species_errors_plflame = evaluate_species_errors(
        current_model, ignition_conditions, psr_conditions, plflame_conditions, initial_metrics_idt, 
        initial_metrics_psr, initial_metrics_plflame, species_limbo, 
        phase_name=phase_name, num_threads=num_threads
        )

    # Use a temporary directory to avoid cluttering the working directory with
    # all the temporary model files
    with TemporaryDirectory() as temp_dir:
        while species_limbo:
            # use difference between error and current error to find species to remove
            aux_err_idt = np.abs(species_errors_idt - current_model.error_idt)
            aux_err_psr = np.abs(species_errors_psr - current_model.error_psr)
            aux_err_plflame = np.abs(species_errors_plflame - current_model.error_plflame)

            print(species_limbo)
            print(aux_err_idt)

            
            idx = np.argmin(aux_err_idt**2 + aux_err_psr**2 + aux_err_plflame**2)
            species_errors_idt = np.delete(species_errors_idt, idx)
            species_errors_psr = np.delete(species_errors_psr, idx)
            species_errors_plflame = np.delete(species_errors_plflame, idx)



            species_remove = species_limbo.pop(idx)
            print('removed:        ', species_remove)

            test_model = trim(
                current_model.filename, [species_remove], f'reduced_model_{species_remove}.yaml', 
                phase_name=phase_name
                )
            #test_model_file = soln2cti.write(
            #    test_model, output_filename=f'reduced_model_{species_remove}.yaml', path=temp_dir
            #    )

            test_model.write_yaml(f'reduced_model_{species_remove}.yaml')
            test_model_file = f'reduced_model_{species_remove}.yaml'

            reduced_model_metrics_idt, reduced_model_metrics_psr , reduced_model_metrics_plflame = sample_metrics(
                test_model_file, ignition_conditions, psr_conditions, plflame_conditions, phase_name=phase_name, 
                num_threads=num_threads, path=path
                )
            error_idt =0
            error_psr = 0
            error_plflame = 0


            if ignition_conditions:
                error_idt = calculate_error(initial_metrics_idt, reduced_model_metrics_idt)

            if psr_conditions:
                error_psr = calculate_error_psr(initial_metrics_psr, reduced_model_metrics_psr)

            if plflame_conditions:
                error_plflame = calculate_error_plflame(initial_metrics_plflame, reduced_model_metrics_plflame)


            logging.info(f'{test_model.n_species:^17} | {species_remove:^17} | {error_idt:^17.2f} | {error_psr:^17.2f} | {error_plflame:^17.2f}')

            # Ensure new error isn't too high
            if error_idt > error_limit or error_psr > error_limit or error_plflame > error_limit:
                break
            else:
                current_model = ReducedModel(model=test_model, filename=test_model_file, error_idt=error_idt, error_psr=error_psr, error_plflame=error_plflame)

            # If using the greedy algorithm, now need to reevaluate all species errors
            if algorithm_type == 'greedy':
                species_errors_idt, species_errors_psr, species_errors_plflame = evaluate_species_errors(
                    current_model, ignition_conditions, psr_conditions, plflame_conditions, 
                    initial_metrics_idt, initial_metrics_psr, initial_metrics_plflame, species_limbo, 
                    phase_name=phase_name, num_threads=num_threads
                    )
                if min(species_errors_idt) > error_limit or min(species_errors_psr) > error_limit or min(species_errors_plflame) > error_limit:
                    break
    
    # Final model; may need to rewrite
    reduced_model = ReducedModel(
        model=current_model.model, filename=f'reduced_{current_model.model.n_species}', 
        error_idt=current_model.error_idt, error_psr=current_model.error_psr, error_plflame=current_model.error_plflame
        )
    #soln2cti.write(reduced_model.model, reduced_model.filename, path=path)
    reduced_model.model.write_yaml(f'reduced_{reduced_model.model.n_species}.yaml')

 

    logging.info(53 * '-')
    logging.info('Sensitivity analysis stage complete.')
    logging.info(f'Skeletal model: {reduced_model.model.n_species} species and '
                 f'{reduced_model.model.n_reactions} reactions.'
                 )
    logging.info(f'Maximum error idt: {reduced_model.error_idt:.2f}%')
    logging.info(f'Maximum error psr: {reduced_model.error_psr:.2f}%')
    logging.info(f'Maximum error plflame: {reduced_model.error_plflame:.2f}%')


    return reduced_model