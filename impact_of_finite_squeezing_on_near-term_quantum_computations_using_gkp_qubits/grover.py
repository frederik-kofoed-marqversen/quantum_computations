import sys
import os
import logging
import json
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import phd_sandbox.dv_circuits as ccs

from simulators.dv_simulator import numpy_quantum as npq
from simulators.dv_simulator import gates as dv_gates
from simulators.dv_simulator.states import State as DVState

from simulators.cv_simulator.simulator import format_time

from simulators.gkp_simulator.transpiler import MBGKPCircuit, parse_to_mps, MPS
from simulators.gkp_simulator.simulator import Simulator as GKPSimulator
from simulators.gkp_simulator.utils import full_logical_density_mps, db2eps

logging.getLogger('simulators').setLevel(logging.WARNING)

def syndrome_matrix(syndromes: list[tuple[int, int]]) -> np.ndarray:
    ms = []
    for x, z in syndromes:
        m = npq.IDTY
        if x:
            m = npq.X @ m
        if z:
            m = npq.Z @ m
        ms.append(m)
    return npq.tensor(*ms)

def grover(tagged: list[int]) -> tuple[list[dv_gates.Gate], list[DVState]]:
    # Grover circuit
    circuit = ccs.grover(ccs.oracle(tagged))
    circuit = circuit[3:]  # First three are Insert(Zero)
    init = [DVState.ZERO] * 3
    # Replace CX with H CZ H
    temp = []
    for gate in circuit:
        if isinstance(gate, dv_gates.CX):
            temp.append(dv_gates.H(gate.target))
            temp.append(dv_gates.CZ(*gate.indices))
            temp.append(dv_gates.H(gate.target))
        else:
            temp.append(gate)
    circuit = temp

    return circuit, init

def test() -> tuple[list[dv_gates.Gate], list[DVState]]:
    circuit = [
        dv_gates.P(0),
        dv_gates.H(1),
        dv_gates.X(0),
        dv_gates.Z(0),
        dv_gates.T(0),
        dv_gates.T(1),
        dv_gates.CZ(0, 1),
        dv_gates.H(0),
        dv_gates.H(1)
    ]
    init = [DVState.H, DVState.H]

    return circuit, init

def run_simulation(simulator: GKPSimulator, init: MPS) -> np.ndarray:
    # Run simulation
    mps, syndromes = simulator.run(init.copy())
    
    # Compute probabilities
    rho = full_logical_density_mps(mps)
    correction = syndrome_matrix(syndromes)
    rho = correction @ rho @ correction.T
    
    return rho

def main():
    # Choose circuit and squeezing levels
    # circuit, init = test()
    # dbs = [7, 10]
    circuit, init = grover([2, 7])
    dbs = np.linspace(5, 15, 13)[2:]
    dbs = np.tile(dbs, 20)

    # Files to write to
    log_file = "test.log"
    data_file = 'test.dat'
    
    # Setup of writables
    log_file = SCRIPT_DIR + "/data/" + log_file
    data_file = SCRIPT_DIR + "/data/" + data_file
    if os.path.exists(data_file):
        raise FileExistsError(f"File {data_file} already exists!")
    open(data_file, 'w').close()
    if log_file is not None:
        open(log_file, 'w').close()
        logging.basicConfig(level=logging.INFO, filename=log_file)
    logger = logging.getLogger(__name__)
    
    # Fixed simulation parameters
    rng = np.random.default_rng(42)
    qs = np.linspace(-20, 20, 1000)    
    svd_options = {
        "rel_err": 1e-2,
        "max_bond_dim": 100,
    }

    # Setup simulator
    gkp_circuit = MBGKPCircuit.transpile(circuit)
    gkp_circuit.fill()
    simulator = GKPSimulator(gkp_circuit, ancilla_epsilon=None, rng_seed=rng, svd_options=svd_options)
    # Gather data
    data = []
    for i, db in tqdm(enumerate(dbs), total=len(dbs), smoothing=0.0):
        logger.info(f"Now starting MB GKP simulation {i+1} of {len(dbs)} with parameter: {db} dB")
        
        # Set simulation parameters
        eps = db2eps(db)
        simulator._epsilon = eps
        
        # Run simulation
        t0 = timer()
        rho = run_simulation(simulator, parse_to_mps(init, eps, qs))
        t1 = timer()

        # Record results
        results = {
            "epsilon": eps,
            "rho_real": rho.real.tolist(),
            "rho_imag": rho.imag.tolist(),
            # "simulation_time": t1 - t0,
            # "rng_state": simulator._rng.bit_generator.state,
        }
        data.append(results)
        
        # Write data to file
        with open(data_file, 'w') as file:
            file.write(json.dumps(data))
        
        logger.info(f"Finished in time: {format_time(t1 - t0)}")

if __name__ == "__main__":
    main()