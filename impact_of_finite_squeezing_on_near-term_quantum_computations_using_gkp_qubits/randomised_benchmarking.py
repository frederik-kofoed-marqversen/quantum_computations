import sys
import os
import logging 
import json
from collections.abc import Sequence
from timeit import default_timer as timer
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from grover import run_simulation

from simulators.dv_simulator import gates as dv_gates
from simulators.dv_simulator.states import State as DVState
from simulators.dv_simulator.simulator import Simulator as DVSimulator
from simulators.dv_simulator import numpy_quantum as npq

from simulators.cv_simulator.simulator import format_time

from simulators.gkp_simulator.simulator import Simulator as GKPSimulator
from simulators.gkp_simulator.transpiler import MBGKPCircuit, parse_to_mps
from simulators.gkp_simulator.utils import db2eps, eps2db

logging.getLogger('simulators').setLevel(logging.WARNING)

gate_list = (dv_gates.I, dv_gates.H, dv_gates.P, dv_gates.Pdg, dv_gates.CZ, dv_gates.SWAP)

def random_circ(N: int, depth: int, rng_seed: int) -> list[dv_gates.Gate]:
    if N < 2:
        raise ValueError("At least 2 qubits required!")
    
    rng = np.random.default_rng(rng_seed)
    indices = list(range(N))
    dv_circ = []
    gkp_circ = MBGKPCircuit(N)
    while gkp_circ.depth() < depth:
        gate = rng.choice(gate_list, 1)[0]
        if issubclass(gate, dv_gates.SingleQubitGate):
            i = int(rng.choice(indices, 1)[0])
            dv_circ.append(gate(i))
            gkp_circ.add_gate(gate(i))
        elif issubclass(gate, dv_gates.TwoQubitGate):
            i = int(rng.choice(indices[-1], 1)[0])
            j = i + 1
            dv_circ.append(gate(i, j))
            gkp_circ.add_gate(gate(i, j))
    gkp_circ.fill()
    return dv_circ, gkp_circ


def sample_depth(db: float, depth: int, num_samples: int, rng_seed: int):
    N = 2
    epsilon = db2eps(db)
    qs = np.linspace(-20, 20, 1000)
    svd_options = {
        "rel_err": 1e-2,
        "max_bond_dim": 100,
    }
    rng = np.random.default_rng(rng_seed)
    init_dv = [DVState.ZERO]*N
    init_mps = parse_to_mps(init_dv, epsilon, qs)

    samples = []
    for _ in range(num_samples):
        dv_circ, gkp_circ = random_circ(N, depth, rng)

        sim = GKPSimulator(gkp_circ, epsilon, rng_seed=rng, svd_options=svd_options)
        rho = run_simulation(sim, init_mps.copy())
        success = DVSimulator(dv_circ).run(init_dv)
        
        fidelity = npq.fidelity(rho, success)
        purity = np.trace(rho @ rho).real
        sample = {"db": db, "depth": depth, "fidelity": fidelity, "purity": purity}
        samples.append(sample)
    return samples

def main():
    # Run script as: OMP_NUM_THREADS=N python3 script.py
    # to make numpy only use N threads. Important to avoid resource fighting
    # when running the script in multiple threads.
    dbs = np.linspace(5, 15, 13)
    dbs = dbs[1:4]
    dbs = np.tile(dbs, 10)
    depths = [8, 10, 15, 15, 20, 20, 20, 20]
    num_samples = 10

    rng = np.random.default_rng()

    # Files to write to
    log_file = "gkp_rb.log"
    data_file = 'gkp_rb.dat'
    
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

    # Gather data
    data = []
    if not isinstance(num_samples, Sequence):
        num_samples = [num_samples]*len(depths)
    for i, db in enumerate(dbs):
        logger.info(f"Now starting RB with squeezing parameter: {db} dB")
        t0 = timer()
        for depth, num in zip(depths, num_samples, strict=True):
            # Run circuits
            logger.info(f"Now sampling circuit depth {depth} a total of {num} times.")
            t1 = timer()
            data += sample_depth(db, int(depth), num, rng)
            t2 = timer()
            logger.info(f"Finished sampling circuit depth {depth} in time: {format_time(t2 - t1)}")
            # Write data to file
            with open(data_file, 'w') as file:
                file.write(json.dumps(data))
        t3 = timer()
        logger.info(f"Finished RB with squeezing parameter: {db} dB in time: {format_time(t3 - t0)}")

if __name__ == "__main__":
    main()