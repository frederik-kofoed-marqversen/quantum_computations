import os
from multiprocessing import Pool
import json
from tqdm import tqdm
import numpy as np
from functools import partial
from itertools import product
import mpmath
mpmath.mp.dps = 24

from utils import surface_code_size
from sequence_class import LogicalDistillationSequence, InitStage, GrowStage
from sequence_optimisation import dfs_code_sequence, DFSArgs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class JobStaticArgs:
    def __init__(self, in_error: float, targ_error: float, code_size_step_size: int, *, no_growing: bool=False):
        # Store given parameters
        self.in_error = mpmath.mpf(in_error)
        self.targ_error = mpmath.mpf(targ_error)

        # Fixed hardware numbers
        # local_gate_time = mpmath.mpf("200e-6")
        local_error = mpmath.mpf("0.1e-2")
        targ_L = surface_code_size(local_error, targ_error)
        # input_rate = mpmath.mpf("100e6") * local_gate_time  # rate in units of gate rate
        L_inj = 3
        inj_rej = mpmath.mpf("8e-2")
        inj_rej = 1 - (1 - inj_rej)**2

        code_sizes = list(range(0, targ_L, code_size_step_size))

        init_seq = LogicalDistillationSequence(InitStage(in_error, L_inj, local_error))
        if no_growing:
            init_seq.add_stage(GrowStage(targ_L, init_seq.L, init_seq.p_L, local_error))
        
        # Store sequence optimisation args
        self.dfs_args = DFSArgs(local_error, 0, targ_error, targ_L, 0, code_sizes=code_sizes)
        self.init_seq = init_seq


def job(args, static_args: JobStaticArgs) -> list[dict]:
        memory, input_rate = args

        dfs_args = static_args.dfs_args.shallow_copy()
        dfs_args.memory = memory
        init_seq = static_args.init_seq.shallow_copy()

        dfs_args.input_rate = input_rate
        seq = dfs_code_sequence(dfs_args, init_seq, 7e-3, print_progress=False)

        return [{
            "memory": int(memory),
            "input_rate": str(input_rate),
            "sequence": seq.serialise() if seq else None
        }]


def main():
    # Static job arguments
    in_error, targ_error = mpmath.mpf("1.25e-2"), mpmath.mpf("1e-12")  # p_bell = 1%, p_targ = 1e-12
    # in_error, targ_error = mpmath.mpf("1.25e-2"), mpmath.mpf("1e-6")  # p_bell = 1%, p_targ = 1e-6    
    # in_error, targ_error = mpmath.mpf("5.2e-2"), mpmath.mpf("1e-6")  # p_bell = 5%, p_targ = 1e-6
    
    no_growing = False
    code_size_step_size = 5
    # code_size_step_size = 1

    # Parameter space: M x r
    memory_arr = [15000]
    rate_arr = np.logspace(np.log10(0.14), np.log10(0.73), 100)

    # File to write to
    data_file_rel_path = 'data/sequences_12_M15000.dat'
    
    # Parallelisation details
    num_threads = 3
    chunksize = 1
    write_time = 10
    
    # Setup writable
    file_path = SCRIPT_DIR + '/' + data_file_rel_path
    if os.path.exists(file_path):
        print(f"File {file_path} already exists! Contents will be overwritten.")
        if input("Do You Want To Continue? [y/n]: ") != "y":
            print("...Exiting")
            exit()
    open(file_path, 'w').close()
    
    # Job wrapping for parallelisation
    static_args = JobStaticArgs(in_error, targ_error, code_size_step_size, no_growing=no_growing)
    job_parallel = partial(job, static_args=static_args)
    
    # Data structure to store results
    data = []
    
    # Parrallelise
    with Pool(num_threads) as pool:
        # Parallelised iterator
        iterator = pool.imap_unordered(job_parallel, product(memory_arr, rate_arr), chunksize=chunksize)
        for results in tqdm(iterator, total=len(memory_arr)*len(rate_arr)):
            # Record results
            data += results
            # Write to disk once in a while
            if len(data) % write_time == 0:
                with open(file_path, 'w') as file:
                    file.write(json.dumps(data))

    # Make sure all data is written to disk
    with open(file_path, 'w') as file:
        file.write(json.dumps(data))


if __name__ == "__main__":
    main()