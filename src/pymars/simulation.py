"""Autoignition simulation module

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

# Standard libraries
import os
import logging

# Related modules
import numpy as np
import tables
import cantera as ct


ct.suppress_thermo_warnings()

# ==========================================================================
#  ___ ____ _____ 
# |_ _|  _ \_   _|
#  | || | | || |  
#  | || |_| || |  
# |___|____/ |_|  
                 

class Simulation(object):
    """Class for ignition delay simulations

    Parameters
    ----------
    idx : int
        Identifer index for case
    properties : InputIgnition
        Object with initial conditions for simulation
    model : str
        Filename for Cantera-format model to be used
    phase_name : str, optional
        Optional name for phase to load from CTI file (e.g., 'gas'). 
    path : str, optional
        Path for location of output files
        
    """
    def __init__(self, idx, properties, model, phase_name='', path=''):
        self.idx = idx
        self.properties = properties
        self.model = model
        self.phase_name = phase_name
        self.path = path

    def setup_case(self):
        """Initialize simulation case.
        """
        self.gas = ct.Solution(self.model, self.phase_name)

        # Default maximum number of steps
        self.max_steps = 10000
        if self.properties.max_steps:
            self.max_steps = self.properties.max_steps

        # By default, simulations will run to steady state, with the maximum number of steps
        # given by ``self.max_steps``. Alternatively, an end time (in seconds) can be
        # given in cases where something specific is needed (e.g., longer than normal)
        self.time_end = 0.0
        if self.properties.end_time:
            self.time_end = self.properties.end_time
    
        self.gas.TP = (
            self.properties.temperature, self.properties.pressure * ct.one_atm
            )
        # set initial composition using either equivalence ratio or general reactant composition
        if self.properties.equivalence_ratio:
            self.gas.set_equivalence_ratio(
                self.properties.equivalence_ratio,
                self.properties.fuel,
                self.properties.oxidizer
                )
        else:
            if self.properties.composition_type == 'mole':
                self.gas.TPX = (
                    self.properties.temperature, self.properties.pressure * ct.one_atm, 
                    self.properties.reactants
                    )
            else:
                self.gas.TPY = (
                    self.properties.temperature, self.properties.pressure * ct.one_atm, 
                    self.properties.reactants
                    )

        if self.properties.kind == 'constant pressure':
            self.reac = ct.IdealGasConstPressureReactor(self.gas)
        else:
            self.reac = ct.IdealGasReactor(self.gas)

        # Create ``ReactorNet`` newtork
        self.sim = ct.ReactorNet([self.reac])

        # Set file for later data file
        self.save_file = os.path.join(self.path, str(self.idx) + '.h5')
        self.sample_points = []

        self.ignition_delay = 0.0

    def run_case(self, stop_at_ignition=False, restart=False):
        """Run simulation case set up ``setup_case``.

        If no end time is specified for the integration, the function integrates
        to steady state (or a maximum of 10,000 steps, by default). This is done
        by checking whether the system state changes below a certain threshold,
        with the residual computed using feature checking. This is blatantly stolen
        from Cantera's :meth:`cantera.ReactorNet.advance_to_steady_state` method.

        Parameters
        ----------
        stop_at_ignition : bool
            If ``True``, stop integration at ignition point, don't save data.
        restart : bool
            If ``True``, skip if results file exists.
        
        Returns
        -------
        self.ignition_delay : float
            Computed ignition delay in seconds

        """

        if restart and os.path.isfile(self.save_file):
            print('Skipped existing case ', self.idx)
            return

        # Save simulation results in hdf5 table format.
        table_def = {'time': tables.Float64Col(pos=0),
                     'temperature': tables.Float64Col(pos=1),
                     'pressure': tables.Float64Col(pos=2),
                     'mass_fractions': tables.Float64Col(
                          shape=(self.reac.thermo.n_species), pos=3
                          ),
                     }

        with tables.open_file(self.save_file, mode='w',
                              title=str(self.idx)
                              ) as h5file:

            table = h5file.create_table(where=h5file.root,
                                        name='simulation',
                                        description=table_def
                                        )
            # Row instance to save timestep information to
            timestep = table.row
            # Save initial conditions
            timestep['time'] = self.sim.time
            timestep['temperature'] = self.reac.T
            timestep['pressure'] = self.reac.thermo.P
            timestep['mass_fractions'] = self.reac.Y
            # Add ``timestep`` to table
            timestep.append()

            ignition_flag = False

            # Main time integration loop
            if self.time_end:
                # if end time specified, continue integration until reaching that time
                while self.sim.time < self.time_end:
                    self.sim.step()

                    # Save new timestep information
                    timestep['time'] = self.sim.time
                    timestep['temperature'] = self.reac.T
                    timestep['pressure'] = self.reac.thermo.P
                    timestep['mass_fractions'] = self.reac.Y

                    if self.reac.T >= self.properties.temperature + 400.0 and not ignition_flag:
                        self.ignition_delay = self.sim.time
                        ignition_flag = True

                        if stop_at_ignition:
                            break

                    # Add ``timestep`` to table
                    timestep.append()
                
            else:
                # otherwise, integrate until steady state, or maximum number of steps reached
                self.sim.reinitialize()
                max_state_values = self.sim.get_state()
                residual_threshold = 10. * self.sim.rtol
                absolute_tolerance = self.sim.atol

                for step in range(self.max_steps):
                    previous_state = self.sim.get_state()

                    self.sim.step()

                    # Save new timestep information
                    timestep['time'] = self.sim.time
                    timestep['temperature'] = self.reac.T
                    timestep['pressure'] = self.reac.thermo.P
                    timestep['mass_fractions'] = self.reac.Y

                    if self.reac.T >= self.properties.temperature + 400.0 and not ignition_flag:
                        self.ignition_delay = self.sim.time
                        ignition_flag = True

                        if stop_at_ignition:
                            break

                    # Add ``timestep`` to table
                    timestep.append()

                    state = self.sim.get_state()
                    max_state_values = np.maximum(max_state_values, state)
                    residual = np.linalg.norm(
                        (state - previous_state) / (max_state_values + absolute_tolerance)
                        ) / np.sqrt(self.sim.n_vars)

                    if residual < residual_threshold:
                        break
                
                if step == self.max_steps - 1:
                    logging.error(
                        'Maximum number of steps reached before '
                        f'convergence for ignition case {self.idx}'
                        )
                    raise RuntimeError(
                        'Maximum number of steps reached before '
                        f'convergence for ignition case {self.idx}'
                    )

            # Write ``table`` to disk
            table.flush()

            if not ignition_flag:
                logging.error(f'No ignition detected for ignition case {self.idx}')
                raise RuntimeError(f'No ignition detected for ignition case {self.idx}')

        return self.ignition_delay

    def calculate_ignition(self):

        #print('inside calc_ig t, T, P, phi, model :    ', self.sim.time, self.reac.T, self.reac.thermo.P, self.properties.equivalence_ratio, self.model)
        
        """Run simulation case set up ``setup_case``, just for ignition delay.
        """        
        # Main time integration loop
        if self.time_end:
            # if end time specified, continue integration until reaching that time
            while self.sim.time < self.time_end:
                self.sim.step()
                if self.reac.T >= self.properties.temperature + 400.0:
                    self.ignition_delay = self.sim.time
                    break
            if not self.ignition_delay:
                logging.warning(
                    f'No ignition detected before end time for ignition case {self.idx}'
                    )
        else:
            # otherwise, integrate until steady state, or maximum number of steps reached
            for step in range(self.max_steps):
                self.sim.step()
                if self.reac.T >= self.properties.temperature + 400.0:
                    self.ignition_delay = self.sim.time
                    break
            if step == self.max_steps - 1:
                logging.warning(
                    'Maximum number of steps reached before '
                    f'convergence for ignition case {self.idx}'
                    )

        return self.ignition_delay

    def process_results(self, skip_data=False):
        """Process integration results to sample data

        Parameters
        ----------
        skip_data : bool
            Flag to skip sampling thermochemical data

        Returns
        -------
        tuple of float, numpy.ndarray or float
            Ignition delay, or ignition delay and sampled data

        """
        delta = 0.05
        deltas = np.arange(delta, 1 + delta, delta)

        # Load saved integration results
        self.save_file = os.path.join(self.path, str(self.idx) + '.h5')
        with tables.open_file(self.save_file, 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation

            times = table.col('time')
            temperatures = table.col('temperature')
            pressures = table.col('pressure')
            mass_fractions = table.col('mass_fractions')

        temperature_initial = temperatures[0]
        temperature_max = temperatures[len(temperatures)-1]
        temperature_diff = temperature_max - temperature_initial 

        sampled_data = np.zeros((len(deltas), 2 + mass_fractions.shape[1]))

        # need to add processing to get the 20 data points here
        self.ignition_delay = 0.0
        ignition_flag = False
        idx = 0
        for time, temp, pres, mass in zip(
            times, temperatures, pressures, mass_fractions
            ):
            if temp >= temperature_initial + 400.0 and not ignition_flag:
                    self.ignition_delay = time
                    ignition_flag = True
                    if skip_data:
                        return self.ignition_delay
            
            if temp >= temperature_initial + (deltas[idx] * temperature_diff):
                sampled_data[idx, 0:2] = [temp, pres]
                sampled_data[idx, 2:] = mass

                idx += 1
                if idx == 20:
                    self.sampled_data = sampled_data
                    return self.ignition_delay, sampled_data

    def clean(self):
        """Delete HDF5 file with full integration data.
        """
        try:
            os.remove(self.save_file)
        except OSError:
            pass


# ==========================================================================
#  ____  ____  ____  
# |  _ \/ ___||  _ \ 
# | |_) \___ \| |_) |
# |  __/ ___) |  _ < 
# |_|   |____/|_| \_\
 


class Simulation_PSR(object):
    """Class for PSR simulations

    Parameters
    ----------
    idx : int
        Identifer index for case
    properties : InputPSR
        Object with initial conditions for simulation
    model : str
        Filename for Cantera-format model to be used
    phase_name : str, optional
        Optional name for phase to load from yaml file (e.g., 'gas'). 
    path : str, optional
        Path for location of output files
        
    """
    def __init__(self, idx, properties, model, phase_name='', path=''):
        self.idx = idx
        self.properties = properties
        self.model = model
        self.phase_name = phase_name
        self.path = path

    def setup_case(self):
        """Initialize simulation case.
        """
        self.gas = ct.Solution(self.model, self.phase_name)

        # Default maximum number of steps
        #self.max_steps = 10000
        #if self.properties.max_steps:
        #    self.max_steps = self.properties.max_steps

        # By default, simulations will run to steady state, with the maximum number of steps
        # given by ``self.max_steps``. Alternatively, an end time (in seconds) can be
        # given in cases where something specific is needed (e.g., longer than normal)
        self.time_end = 0.0
        if self.properties.end_time:
            self.time_end = self.properties.end_time
    
        self.gas.TP = (
            self.properties.temperature, self.properties.pressure * ct.one_atm
            )
        # set initial composition using either equivalence ratio or general reactant composition
        if self.properties.equivalence_ratio:
            self.gas.set_equivalence_ratio(
                self.properties.equivalence_ratio,
                self.properties.fuel,
                self.properties.oxidizer
                )
        else:
            if self.properties.composition_type == 'mole':
                self.gas.TPX = (
                    self.properties.temperature, self.properties.pressure * ct.one_atm, 
                    self.properties.reactants
                    )
            else:
                self.gas.TPY = (
                    self.properties.temperature, self.properties.pressure * ct.one_atm, 
                    self.properties.reactants
                    )

        #if self.properties.kind == 'constant pressure':
        self.fuel_air_mixture_tank = ct.Reservoir(self.gas)
        self.exhaust = ct.Reservoir(self.gas)
        self.reac = ct.IdealGasConstPressureReactor(self.gas, energy='off')
        
        self.mass_flow_controller = ct.MassFlowController(upstream=self.fuel_air_mixture_tank,downstream=self.reac,mdot=self.reac.mass / self.properties.residence_time)
        self.pressure_regulator   = ct.PressureController(upstream=self.reac, downstream=self.exhaust, primary=self.mass_flow_controller,K=1e-3)

        #else:
        #    self.reac = ct.IdealGasReactor(self.gas)

        # Create ``ReactorNet`` newtork
        self.sim = ct.ReactorNet([self.reac])

        # Set file for later data file
        self.save_file = os.path.join(self.path, str(self.idx) + '.h5')
        self.sample_points = []

        #self.ignition_delay = 0.0

    def run_case(self, restart=False):
        """Run simulation case set up ``setup_case``.

        If no end time is specified for the integration, the function integrates
        to steady state (or a maximum of 10,000 steps, by default). This is done
        by checking whether the system state changes below a certain threshold,
        with the residual computed using feature checking. This is blatantly stolen
        from Cantera's :meth:`cantera.ReactorNet.advance_to_steady_state` method.

        Parameters
        ----------
        stop_at_ignition : bool
            If ``True``, stop integration at ignition point, don't save data.
        restart : bool
            If ``True``, skip if results file exists.
        
        Returns
        -------
        self.ignition_delay : float
            Computed ignition delay in seconds

        """

        if restart and os.path.isfile(self.save_file):
            print('Skipped existing case ', self.idx)
            return

        # Save simulation results in hdf5 table format.
        table_def = {'time': tables.Float64Col(pos=0),
                     'temperature': tables.Float64Col(pos=1),
                     'pressure': tables.Float64Col(pos=2),
                     'mass_fractions': tables.Float64Col(
                          shape=(self.reac.thermo.n_species), pos=3
                          ),
                     }

        with tables.open_file(self.save_file, mode='w',
                              title=str(self.idx)
                              ) as h5file:

            table = h5file.create_table(where=h5file.root,
                                        name='simulation',
                                        description=table_def
                                        )
            # Row instance to save timestep information to
            timestep = table.row
            # Save initial conditions
            timestep['time'] = self.sim.time
            timestep['temperature'] = self.reac.T
            timestep['pressure'] = self.reac.thermo.P
            timestep['mass_fractions'] = self.reac.Y
            # Add ``timestep`` to table
            timestep.append()

            #ignition_flag = False

            # Main time integration loop
            #if self.time_end:
                # if end time specified, continue integration until reaching that time
            while self.sim.time < self.time_end:
                    self.sim.step()

                    # Save new timestep information
                    timestep['time'] = self.sim.time
                    timestep['temperature'] = self.reac.T
                    timestep['pressure'] = self.reac.thermo.P
                    timestep['mass_fractions'] = self.reac.Y

                    #if self.reac.T >= self.properties.temperature + 400.0 and not ignition_flag:
                    #    self.ignition_delay = self.sim.time
                    #    ignition_flag = True
                    #
                    #    if stop_at_ignition:
                    #        break

                    # Add ``timestep`` to table
                    timestep.append()
                
           # else:
                # otherwise, integrate until steady state, or maximum number of steps reached
            #    self.sim.reinitialize()
            #    max_state_values = self.sim.get_state()
            #    residual_threshold = 10. * self.sim.rtol
            #    absolute_tolerance = self.sim.atol

            #    for step in range(self.max_steps):
            #        previous_state = self.sim.get_state()

            #        self.sim.step()

                    # Save new timestep information
            #        timestep['time'] = self.sim.time
            #        timestep['temperature'] = self.reac.T
            #        timestep['pressure'] = self.reac.thermo.P
            #        timestep['mass_fractions'] = self.reac.Y

            #        if self.reac.T >= self.properties.temperature + 400.0 and not ignition_flag:
            #            self.ignition_delay = self.sim.time
            #            ignition_flag = True

            #           if stop_at_ignition:
            #                break

                    # Add ``timestep`` to table
            #        timestep.append()

            #        state = self.sim.get_state()
            #        max_state_values = np.maximum(max_state_values, state)
            #        residual = np.linalg.norm(
            #            (state - previous_state) / (max_state_values + absolute_tolerance)
            #            ) / np.sqrt(self.sim.n_vars)

            #       if residual < residual_threshold:
            #            break
                
            #    if step == self.max_steps - 1:
            #        logging.error(
            #            'Maximum number of steps reached before '
            #            f'convergence for ignition case {self.idx}'
            #            )
            #        raise RuntimeError(
            #            'Maximum number of steps reached before '
            #            f'convergence for ignition case {self.idx}'
            #        )

            # Write ``table`` to disk
            table.flush()

            #if not ignition_flag:
            #    logging.error(f'No ignition detected for ignition case {self.idx}')
            #    raise RuntimeError(f'No ignition detected for ignition case {self.idx}')

        #return self.ignition_delay
        X_PSR_vec = np.zeros(len(self.properties.target_species_PSR))
        ii_aux = 0
        for species in self.properties.target_species_PSR:
             idx_aux = self.gas.species_names.index(species)
             X_PSR_vec[ii_aux] = self.gas.X[idx_aux]
             ii_aux = ii_aux +1
             
        
        return X_PSR_vec

    def calculate_psr(self):
        """Run simulation case set up ``setup_case``.
        """        
        # Main time integration loop
        #if self.time_end:
            # if end time specified, continue integration until reaching that time
        while self.sim.time < self.time_end:
               #print('timeeeee:      ', self.sim.time, self.time_end)
               self.sim.step()
               #aux = self.sim.time

               #print('timeeeee:      ', self.sim.time, self.time_end)


                #if self.reac.T >= self.properties.temperature + 400.0:
                #    self.ignition_delay = self.sim.time
                #    break
        #if not self.ignition_delay:
        #        logging.warning(
        #            f'No ignition detected before end time for ignition case {self.idx}'
        #            )
       # else:
       #     # otherwise, integrate until steady state, or maximum number of steps reached
       #     for step in range(self.max_steps):
       #         self.sim.step()
       #         if self.reac.T >= self.properties.temperature + 400.0:
       #             self.ignition_delay = self.sim.time
       #             break
       #     if step == self.max_steps - 1:
       #         logging.warning(
       #             'Maximum number of steps reached before '
       #             f'convergence for ignition case {self.idx}'
       #             )

        #return self.ignition_delay
        X_PSR_vec = np.zeros(len(self.properties.target_species_PSR))
        ii_aux = 0
        for species in self.properties.target_species_PSR:
             idx_aux = self.gas.species_names.index(species)
             X_PSR_vec[ii_aux] = self.gas.X[idx_aux]
             ii_aux = ii_aux +1

        #print('X inside calc_psr:   ', X_PSR_vec, self.time_end, aux)    

             
             
        
        return X_PSR_vec

    def process_results(self, skip_data=False):
        """Process integration results to sample data

        Parameters
        ----------
        skip_data : bool
            Flag to skip sampling thermochemical data

        Returns
        -------
        tuple of float, numpy.ndarray or float
            Ignition delay, or ignition delay and sampled data

        """
        delta = 0.05
        deltas = np.arange(delta, 1 + delta, delta)

        # Load saved integration results
        self.save_file = os.path.join(self.path, str(self.idx) + '.h5')
        with tables.open_file(self.save_file, 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation

            times = table.col('time')
            temperatures = table.col('temperature')
            pressures = table.col('pressure')
            mass_fractions = table.col('mass_fractions')

        #temperature_initial = temperatures[0]
        #temperature_max = temperatures[len(temperatures)-1]
        #temperature_diff = temperature_max - temperature_initial 

        
        #sampled_data = np.zeros((len(deltas), 2 + mass_fractions.shape[1]))
        sampled_data = np.zeros((1, 2 + mass_fractions.shape[1]))
        #print('sampled_data:', sampled_data)
        

        # need to add processing to get the 20 data points here
        #self.ignition_delay = 0.0
        #ignition_flag = False
        idx = 0
        #for time, temp, pres, mass in zip(
        #    times, temperatures, pressures, mass_fractions
            #):
            #if temp >= temperature_initial + 400.0 and not ignition_flag:
            #        self.ignition_delay = time
            #        ignition_flag = True
            #        if skip_data:
            #            return self.ignition_delay
            
         #   if temp >= temperature_initial + (deltas[idx] * temperature_diff):
         #       sampled_data[idx, 0:2] = [temp, pres]
         #       sampled_data[idx, 2:] = mass

         #      idx += 1
         #       if idx == 20:
        temp = temperatures[-1] 
        pres = pressures[-1]
        mass = mass_fractions[-1] 
        sampled_data[idx, 0:2] = [temp, pres]
        sampled_data[idx, 2:] = mass

        sol_aux = ct.Solution(self.model)
        sol_aux.TPY = temp, pres, mass

        X_PSR_vec = np.zeros((1,len(self.properties.target_species_PSR)))
        ii_aux = 0
        for species in self.properties.target_species_PSR:
             idx_aux = sol_aux.species_names.index(species)
             X_PSR_vec[0,ii_aux] =sol_aux.X[idx_aux]
             ii_aux = ii_aux +1
             
         
        self.sampled_data = sampled_data
        return  X_PSR_vec, sampled_data

    def clean(self):
        """Delete HDF5 file with full integration data.
        """
        try:
            os.remove(self.save_file)
        except OSError:
            pass



# ==========================================================================
#  _     ______     __
# | |   | __ ) \   / /
# | |   |  _ \\ \ / / 
# | |___| |_) |\ V /  
# |_____|____/  \_/   

# ==========================================================================



class Simulation_PLFlame(object):
    """Class for Premixed Laminar Flame simulations

    Parameters
    ----------
    idx : int
        Identifer index for case
    properties : InputPSR
        Object with initial conditions for simulation
    model : str
        Filename for Cantera-format model to be used
    phase_name : str, optional
        Optional name for phase to load from yaml file (e.g., 'gas'). 
    path : str, optional
        Path for location of output files
        
    """
    def __init__(self, idx, properties, model, phase_name='', path=''):
        self.idx = idx
        self.properties = properties
        self.model = model
        self.phase_name = phase_name
        self.path = path

    def setup_case(self):
        """Initialize simulation case.
        """
        self.gas = ct.Solution(self.model, self.phase_name, transport_model = 'mixture-averaged')

        # Default maximum number of steps
        self.max_steps = 10000
        if self.properties.max_steps:
            self.max_steps = self.properties.max_steps

        # By default, simulations will run to steady state, with the maximum number of steps
        # given by ``self.max_steps``. Alternatively, an end time (in seconds) can be
        # given in cases where something specific is needed (e.g., longer than normal)
        #self.time_end = 0.0
        #if self.properties.end_time:
        #    self.time_end = self.properties.end_time
    
        self.gas.TP = (
            self.properties.temperature, self.properties.pressure * ct.one_atm
            )
        # set initial composition using either equivalence ratio or general reactant composition


        #print('EQUIIIIVVVV       ', self.properties.equivalence_ratio)
        
        if self.properties.equivalence_ratio:
            self.gas.set_equivalence_ratio(
                self.properties.equivalence_ratio,
                self.properties.fuel,
                self.properties.oxidizer
                )
        else:
            if self.properties.composition_type == 'mole':
                self.gas.TPX = (
                    self.properties.temperature, self.properties.pressure * ct.one_atm, 
                    self.properties.reactants
                    )
            else:
                self.gas.TPY = (
                    self.properties.temperature, self.properties.pressure * ct.one_atm, 
                    self.properties.reactants
                    )

        #if self.properties.kind == 'constant pressure':
        #self.fuel_air_mixture_tank = ct.Reservoir(self.gas)
        #self.exhaust = ct.Reservoir(self.gas)
        #self.reac = ct.IdealGasConstPressureReactor(self.gas, energy='off')
        
        #self.mass_flow_controller = ct.MassFlowController(upstream=self.fuel_air_mixture_tank,downstream=self.reac,mdot=self.reac.mass / self.properties.residence_time)
        #self.pressure_regulator   = ct.PressureController(upstream=self.reac, downstream=self.exhaust, primary=self.mass_flow_controller,K=1e-3)

        #else:
        #    self.reac = ct.IdealGasReactor(self.gas)

        # Create ``ReactorNet`` newtork
        #self.sim = ct.ReactorNet([self.reac])

        # Set file for later data file
        self.f = ct.FreeFlame(self.gas, width=0.03)
        #self.f.transport_model = 'mixture-averaged'
        self.f.set_refine_criteria(ratio=3, slope=0.06, curve=0.10)
       
        self.save_file = os.path.join(self.path, str(self.idx) + '.h5')
        self.sample_points = []

        #self.ignition_delay = 0.0

    def run_case(self, restart=False):
        """Run simulation case set up ``setup_case``.

        If no end time is specified for the integration, the function integrates
        to steady state (or a maximum of 10,000 steps, by default). This is done
        by checking whether the system state changes below a certain threshold,
        with the residual computed using feature checking. This is blatantly stolen
        from Cantera's :meth:`cantera.ReactorNet.advance_to_steady_state` method.

        Parameters
        ----------
        stop_at_ignition : bool
            If ``True``, stop integration at ignition point, don't save data.
        restart : bool
            If ``True``, skip if results file exists.
        
        Returns
        -------
        self.ignition_delay : float
            Computed ignition delay in seconds

        """

        if restart and os.path.isfile(self.save_file):
            print('Skipped existing case ', self.idx)
            return

        # Save simulation results in hdf5 table format.
        table_def = {'x': tables.Float64Col(pos=0),
                     'velocity': tables.Float64Col(pos=1),
                     'temperature': tables.Float64Col(pos=2),
                     'pressure': tables.Float64Col(pos=3),
                     'mass_fractions': tables.Float64Col(
                          shape=(self.gas.n_species), pos=4
                          ),
                     }

        try: 
            self.f.solve(loglevel=0, auto=True)

        except ct.CanteraError:
            print('error file:       ', self.model)
            print('sl:       ', self.f.velocity[0]*100)

            


        with tables.open_file(self.save_file, mode='w',
                              title=str(self.idx)
                              ) as h5file:

            table = h5file.create_table(where=h5file.root,
                                        name='simulation',
                                        description=table_def
                                        )
            # Row instance to save spatialstep information to
            spatialstep = table.row

            # Save initial conditions
            for i_aux in range(len(self.f.grid)):
                spatialstep['x'] = self.f.grid[i_aux]
                spatialstep['velocity'] = self.f.velocity[i_aux]*100
                spatialstep['temperature'] = self.f.T[i_aux]
                spatialstep['pressure'] = self.f.P
                spatialstep['mass_fractions'] = self.f.Y[:, i_aux]
                # Add ``timestep`` to table
                spatialstep.append()

            #ignition_flag = False

            # Main time integration loop
            #if self.time_end:
                # if end time specified, continue integration until reaching that time
            #while self.sim.time < self.time_end:
             #       self.sim.step()

                    # Save new timestep information
             #       timestep['time'] = self.sim.time
             #       timestep['temperature'] = self.reac.T
             #       timestep['pressure'] = self.reac.thermo.P
             #       timestep['mass_fractions'] = self.reac.Y

                    #if self.reac.T >= self.properties.temperature + 400.0 and not ignition_flag:
                    #    self.ignition_delay = self.sim.time
                    #    ignition_flag = True
                    #
                    #    if stop_at_ignition:
                    #        break

                    # Add ``timestep`` to table
              #      timestep.append()
                
           # else:
                # otherwise, integrate until steady state, or maximum number of steps reached
            #    self.sim.reinitialize()
            #    max_state_values = self.sim.get_state()
            #    residual_threshold = 10. * self.sim.rtol
            #    absolute_tolerance = self.sim.atol

            #    for step in range(self.max_steps):
            #        previous_state = self.sim.get_state()

            #        self.sim.step()

                    # Save new timestep information
            #        timestep['time'] = self.sim.time
            #        timestep['temperature'] = self.reac.T
            #        timestep['pressure'] = self.reac.thermo.P
            #        timestep['mass_fractions'] = self.reac.Y

            #        if self.reac.T >= self.properties.temperature + 400.0 and not ignition_flag:
            #            self.ignition_delay = self.sim.time
            #            ignition_flag = True

            #           if stop_at_ignition:
            #                break

                    # Add ``timestep`` to table
            #        timestep.append()

            #        state = self.sim.get_state()
            #        max_state_values = np.maximum(max_state_values, state)
            #        residual = np.linalg.norm(
            #            (state - previous_state) / (max_state_values + absolute_tolerance)
            #            ) / np.sqrt(self.sim.n_vars)

            #       if residual < residual_threshold:
            #            break
                
            #    if step == self.max_steps - 1:
            #        logging.error(
            #            'Maximum number of steps reached before '
            #            f'convergence for ignition case {self.idx}'
            #            )
            #        raise RuntimeError(
            #            'Maximum number of steps reached before '
            #            f'convergence for ignition case {self.idx}'
            #        )

            # Write ``table`` to disk
            table.flush()

            #if not ignition_flag:
            #    logging.error(f'No ignition detected for ignition case {self.idx}')
            #    raise RuntimeError(f'No ignition detected for ignition case {self.idx}')

        #return self.ignition_delay
        #plflame_lbv_vec = np.zeros(len(self.properties.target_species_PSR))
        #ii_aux = 0
        #for species in self.properties.target_species_PSR:
        #     idx_aux = self.gas.species_names.index(species)
        #     X_PSR_vec[ii_aux] = self.gas.X[idx_aux]
        #     ii_aux = ii_aux +1

        self.plflame_lbv = self.f.velocity[0] *100 #cm/s

        plflame_lbv = self.plflame_lbv
             
        
        return plflame_lbv

    def calculate_plflame(self):
        """Run simulation case set up ``setup_case``.
        """        
        try: 
            self.f.solve(loglevel=0, auto=True)

        except ct.CanteraError:
            print('error file:       ', self.model)
            print('sl:       ', self.f.velocity[0]*100)

            
        # Main time integration loop
        #if self.time_end:
            # if end time specified, continue integration until reaching that time
            
        #while self.sim.time < self.time_end:
        #       #print('timeeeee:      ', self.sim.time, self.time_end)
        #       self.sim.step()
               #print('timeeeee:      ', self.sim.time, self.time_end)


                #if self.reac.T >= self.properties.temperature + 400.0:
                #    self.ignition_delay = self.sim.time
                #    break
        #if not self.ignition_delay:
        #        logging.warning(
        #            f'No ignition detected before end time for ignition case {self.idx}'
        #            )
       # else:
       #     # otherwise, integrate until steady state, or maximum number of steps reached
       #     for step in range(self.max_steps):
       #         self.sim.step()
       #         if self.reac.T >= self.properties.temperature + 400.0:
       #             self.ignition_delay = self.sim.time
       #             break
       #     if step == self.max_steps - 1:
       #         logging.warning(
       #             'Maximum number of steps reached before '
       #             f'convergence for ignition case {self.idx}'
       #             )

        #return self.ignition_delay
        #X_PSR_vec = np.zeros(len(self.properties.target_species_PSR))
        #ii_aux = 0
        #for species in self.properties.target_species_PSR:
        #     idx_aux = self.gas.species_names.index(species)
        #     X_PSR_vec[ii_aux] = self.gas.X[idx_aux]
        #     ii_aux = ii_aux +1

        self.plflame_lbv = self.f.velocity[0] *100 #cm/s

        plflame_lbv = self.plflame_lbv
             
             
        
        return plflame_lbv

    def process_results(self, skip_data=False):
        """Process integration results to sample data

        Parameters
        ----------
        skip_data : bool
            Flag to skip sampling thermochemical data

        Returns
        -------
        tuple of float, numpy.ndarray or float
            Ignition delay, or ignition delay and sampled data

        """
        delta = 0.05
        deltas = np.arange(delta, 1 + delta, delta)

        # Load saved integration results
        self.save_file = os.path.join(self.path, str(self.idx) + '.h5')
        with tables.open_file(self.save_file, 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation

            x_grid = table.col('x')
            velocities = table.col('velocity')
            temperatures = table.col('temperature')
            pressures = table.col('pressure')
            mass_fractions = table.col('mass_fractions')

        temperature_initial = temperatures[0]
        temperature_max = temperatures[len(temperatures)-1]
        temperature_diff = temperature_max - temperature_initial 

        
        sampled_data = np.zeros((len(deltas), 2 + mass_fractions.shape[1]))
        #sampled_data = np.zeros((1, 2 + mass_fractions.shape[1]))
        #print('sampled_data:', sampled_data)
        

        # need to add processing to get the 20 data points here
        #self.ignition_delay = 0.0
        #ignition_flag = False
        idx = 0
        #for time, temp, pres, mass in zip(
        #    times, temperatures, pressures, mass_fractions
            #):
            #if temp >= temperature_initial + 400.0 and not ignition_flag:
            #        self.ignition_delay = time
            #        ignition_flag = True
            #        if skip_data:
            #            return self.ignition_delay
            
         #   if temp >= temperature_initial + (deltas[idx] * temperature_diff):
         #       sampled_data[idx, 0:2] = [temp, pres]
         #       sampled_data[idx, 2:] = mass

         #      idx += 1
         #       if idx == 20:

        self.plflame_lbv = velocities[0]

        for x, temp, pres, mass in zip(
            x_grid, temperatures, pressures, mass_fractions
            ):
            #if temp >= temperature_initial + 400.0 and not ignition_flag:
            #        self.ignition_delay = time
            #        ignition_flag = True
            #        if skip_data:
            #            return self.ignition_delay
            
            if temp >= temperature_initial + (deltas[idx] * temperature_diff):
                #print('idx, temp:    ', idx, temp)
                sampled_data[idx, 0:2] = [temp, pres]
                sampled_data[idx, 2:] = mass

                idx += 1
                if idx == 20:
                    self.sampled_data = sampled_data
                    return self.plflame_lbv, sampled_data



         


    def clean(self):
        """Delete HDF5 file with full integration data.
        """
        try:
            os.remove(self.save_file)
        except OSError:
            pass


                     
