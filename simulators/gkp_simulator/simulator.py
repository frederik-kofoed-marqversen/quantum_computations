from typing import Callable

import numpy as np
from timeit import default_timer as timer

from ..cv_simulator.simulator import Simulator as CVSimulator, format_time
from ..cv_simulator.gate_abc import SVD_OPTIONS, MPS, MeasurementResult
from ..cv_simulator.gates import F as FourierGate

from ..dv_simulator import gates as dv_gates
from ..dv_simulator.gates import Gate as DVGate

from .gates import SQPI
from .transpiler import MBGKPCircuit, MeasurementBased, gate_transpile, ClassicalControl, Syndrome
from .utils import format_result

import logging
logger = logging.getLogger(__name__)


def measurement_formatter(result: MeasurementResult) -> str:
    return format_result(result.result)


# Commutes `gate` through `paulis` such that `gate * paulis = paulis' * gate'`
def commute(gate: dv_gates.Gate, paulis: list[Syndrome]) -> tuple[list[Syndrome], dv_gates.Gate]:
    paulis = [list(p) for p in paulis]
    match type(gate):
        case dv_gates.I:
            pass
        case dv_gates.T:
            idx = gate.indices[0]
            if paulis[idx][0] == 1:
                gate = dv_gates.Tdg(*gate.indices)
        case dv_gates.Tdg:
            idx = gate.indices[0]
            if paulis[idx][0] == 1:
                gate = dv_gates.T(*gate.indices)
        case dv_gates.H:
            idx = gate.indices[0]
            paulis[idx][0], paulis[idx][1] = paulis[idx][1], paulis[idx][0]
        case dv_gates.P | dv_gates.Pdg:
            idx = gate.indices[0]
            paulis[idx][1] ^= paulis[idx][0]
        case dv_gates.CZ:
            idx1, idx2 = gate.indices
            paulis[idx1][1] ^= paulis[idx2][0]
            paulis[idx2][1] ^= paulis[idx1][0]
        case dv_gates.SWAP:
            idx1, idx2 = gate.indices
            paulis[idx1], paulis[idx2] = paulis[idx2], paulis[idx1]
        case _:
            raise NotImplementedError(f"Commutator logic for gate: {gate} not implemented.")
    paulis = [tuple(p) for p in paulis]
    return paulis, gate


class Simulator(CVSimulator):
    def __init__(
            self, 
            circuit: MBGKPCircuit, 
            ancilla_epsilon: float,
            *,
            rng_seed: int=None, 
            svd_options: dict={},
            debug_info: Callable[['Simulator'], None]=None,
        ):
        """
        A mix of cvsimulator and dvsimulator. We take also a simulation wide
        ancilla epsilon parameter which will be passed to all Insert gates if they do not define one themselves already.
        This means that Insert gates in circuits can be simplified by ignoring the epsilon parameter and just letting
        the simulator pass over `ancilla_epsilon`.
        """
        self._circuit: MBGKPCircuit = circuit
        self._N = circuit._N
        self._rng: np.random.Generator = np.random.default_rng(rng_seed)
        self._epsilon = ancilla_epsilon
        
        self._state: MPS = None
        self.pauli_syndrome: list[Syndrome] = None
        
        svd_options = svd_options.copy()
        self._svd_options = {key: svd_options.pop(key) for key in SVD_OPTIONS if key in svd_options}
        if svd_options:
            logging.warning(f"{type(self).__name__} recieved unexpected keys in svd_options: {svd_options.keys()}")
        
        self.debug_info: Callable[['Simulator'], None] = debug_info or (lambda _: None)

    def apply_gate(self, dv_gate: DVGate) -> tuple[list[Syndrome], list[int]]:
        gate: MeasurementBased = gate_transpile(dv_gate, epsilon=self._epsilon, **self._svd_options)
        sim = CVSimulator(gate.compile(), rng_seed=self._rng, measurement_formatter=measurement_formatter)
        self._state = sim.run(self._state)
        results = [r.result for r in sim.results]
        return gate.compute_syndrome(results)
    
    def apply_paulis(self, paulis: list[Syndrome]):
        for i in range(len(self.pauli_syndrome)):
            s1, s2 = self.pauli_syndrome[i], paulis[i]
            self.pauli_syndrome[i] = (s1[0] ^ s2[0], s1[1] ^ s2[1])
        
    def run(self, initial_state: MPS) -> tuple[MPS, list[Syndrome]]:
        initial_state.validate()
        
        # Reset simulation parameters
        self._state = initial_state
        self.pauli_syndrome: list[Syndrome] = [(0, 0) for _ in range(self._N)]
        gate_syndromes: list[list[Syndrome]] = [[(0, 0)] * self._N] * 2
        
        # Loop over layers in circuit
        circ_start = timer()
        num_layers = len(self._circuit._layers)
        logger.info(f"Total number of MB gates: {self._circuit.count()} in a total of {num_layers} layers.")
        for i, layer in enumerate(self._circuit._layers):
            logger.info(f"Layer {i+1} of {num_layers}.")
            
            # Add space for syndromes from this layer and remove oldest layer syndromes
            gate_syndromes.pop(0)
            gate_syndromes.append([(0, 0)] * self._N)
            # Loop over gates in current layer
            for gate in layer.gates:
                if isinstance(gate, ClassicalControl):
                    # Compute T-gate correction operator
                    if gate_syndromes[-2][gate.indices[0]][0]:
                        gate = gate.gate
                    else:
                        gate = dv_gates.I(*gate.indices)
                
                # Commute gate through current Pauli syndrome
                self.pauli_syndrome, gate = commute(gate, self.pauli_syndrome)
                
                # Apply gate
                logger.info(f"MB gate: {gate}")
                syndromes, indices = self.apply_gate(gate)
                logger.info(f"Gate syndrome: {syndromes}")

                # Record gate syndromes
                for i, s in zip(indices, syndromes, strict=True):
                    gate_syndromes[-1][i] = s
            
            # Apply Pauli operators and syndrome corrections of current layer
            logger.info(f"Applying syndrome correction: {gate_syndromes[-1]}")
            self.apply_paulis(gate_syndromes[-1])
            logger.info(f"Applying Pauli operators: {layer.paulis}")
            self.apply_paulis(layer.paulis)
            logger.info(f"Final Pauli syndrome: {self.pauli_syndrome}")

            if logger.isEnabledFor(logging.DEBUG):
                    self.debug_info(self)
        
        logger.info("Finished MB GKP simulation!")
        logger.info("Total time: " + format_time(timer() - circ_start))
        
        return self._state, [tuple(s) for s in self.pauli_syndrome]


class SimulatorAlt(Simulator):
    def apply_gate(self, dv_gate) -> tuple[list[Syndrome], list[int]]:
        match type(dv_gate):
            case dv_gates.I:
                return [(0, 0)], dv_gate.indices
            case dv_gates.H:
                FourierGate(dv_gate.indices[0]).apply(self._state)
                return [(0, 0)], dv_gate.indices
            case _:
                return super().apply_gate(dv_gate)