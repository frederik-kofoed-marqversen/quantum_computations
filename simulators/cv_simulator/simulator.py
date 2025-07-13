import numpy as np
from timeit import default_timer as timer
from collections.abc import Callable

from .mps import MPS, SVD_OPTIONS
from .gate_abc import Gate, MeasurementResult

import logging
logger = logging.getLogger(__name__)

def format_time(time_in_seconds: float) -> str:
    t = time_in_seconds
    mins = int(np.floor(t // 60))
    t = t % 60
    secs = int(np.floor(t))
    t -= secs
    millies = round(t * 1000)
    return ":".join([str(mins).rjust(2, "0"), str(secs).rjust(2, "0"), str(millies).rjust(3, "0")])

class Simulator:
    def __init__(
            self, 
            gates: list[Gate], 
            rng_seed: int=None, 
            *, 
            debug_info: Callable[['Simulator'], None]=None, 
            measurement_formatter: Callable[[MeasurementResult], str]=None, 
            svd_options: dict={},
        ):
        """
        Initialize the simulator with a list of operations.

        Parameters:
            gates (list[CVGate]): A list of Gate objects representing the quantum operations to simulate.
            rng_seed (int, optional): Seed for the random number generator used in stochastic operations.
                    Defaults to None, which initializes a random state.
            debug_info (Callable[['Simulator'], None], optional): A callback function for providing debugging information 
                    during simulation. Only used if the logging level is set to DEBUG. Defaults to a no-op lambda function.
            measurement_formatter: Callable[[Result], str]: A formatting function applied to measurement results when logging
            svd_options: Applied to gates only if the gate itself does not define each of the following options. 
                - `max_bond_dim` (float): The maximum bond dimension for SVD truncation. Setting this has huge advantages as 
                        it allows for using randomized truncated SVD which is both faster and more memory efficient than 
                        truncating a full SVD
                - `abs_err` (float): The allowed absolute error for SVD truncation.
                - `rel_err` (float): The allowed relative error for SVD truncation.
        """
        self._gates: list[Gate] = gates
        self._state: MPS = None
        self._rng = np.random.default_rng(rng_seed)
        self.results: list[MeasurementResult] = None
        self.debug_info: Callable[['Simulator'], None] = debug_info or (lambda _: None)
        self.meas_format: Callable[[MeasurementResult], str] = measurement_formatter

        svd_options = svd_options.copy()
        self._svd_options = {key: svd_options.pop(key) for key in SVD_OPTIONS if key in svd_options}
        if svd_options:
            logging.warning(f"{type(self).__name__} recieved unexpected keys in svd_options: {svd_options.keys()}")

    def update_gate(self, gate: Gate):
        """This is for updating certain parameters of gates based on those defined simulation wide."""
        # Update the svd options NOT already defined on the gate to match the simulation wide svd options.
        gate.svd_options.update({key: value for key, value in self._svd_options.items() if key not in gate.svd_options})

    def apply_gate(self, gate: Gate):            
        start = timer()
        output = gate.apply(self._state, rng=self._rng)
        end = timer()

        if isinstance(output, MeasurementResult):  # Record result
            self.results.append(output)
            logger.info(f"   measurement result : " + (f"{self.meas_format(output)}" if self.meas_format else str(output)))
        
        logger.info(f"   mps shape: {self._state.shape()}")
        logger.info("   evaluation time : " + format_time(end - start))

        if logger.isEnabledFor(logging.DEBUG):
            self.debug_info(self)

    def run(self, initial_state: MPS) -> MPS:
        """
        Run the simulator with the given initial state and sequence of options.
        
        :param initial_state: Initial state of the simulator.
        :param options_list: List of options dictionaries, one per gate.
        :return: Final state after running all gates.
        """
        
        initial_state.validate()
        
        self._state = initial_state
        self.results = []
        
        circ_start = timer()
        
        logger.info(f"Total number of gates: {len(self._gates)}")
        for i, gate in enumerate(self._gates):  
            logger.info(f"Gate {i}: {gate}")
            self.update_gate(gate)
            # Apply the gate
            self.apply_gate(gate)
        logger.info("Finished!")
        logger.info("Total time: " + format_time(timer() - circ_start))

        return self._state