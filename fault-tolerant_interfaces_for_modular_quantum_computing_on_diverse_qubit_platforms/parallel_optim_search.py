import os
from multiprocessing import Pool, Manager
import json
from tqdm import tqdm
import numpy as np
from functools import partial
import mpmath
mpmath.mp.dps = 24

from utils import surface_code_size
from sequence_class import LogicalDistillationSequence, InitStage, GrowStage
from sequence_optimisation import dfs_code_sequence, DFSArgs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Globally defined stuff for sharing between parallel processes
manager = Manager()
shared = manager.dict()
shared["M"] = 0
shared["seqs"] = (None, None)
lock = manager.Lock()

class JobStaticArgs:
    def __init__(self, in_error: float, targ_error: float, code_size_step_size: int, *, no_growing: bool=False):
        self.shared = shared
        self.lock = lock
        
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


def job(memory, static_args: JobStaticArgs) -> list[dict]:
        with static_args.lock:
            # Is safe since these sequences are never mutated.
            # They are only ever used to evaluate .distillation_rate()
            # Also they are stored in a tuple which cannot be mutated in place.
            prev_seqs = static_args.shared["seqs"]
        
        dfs_args = static_args.dfs_args.shallow_copy()
        dfs_args.memory = memory
        init_seq = static_args.init_seq.shallow_copy()

        seqs = [None, None]
        input_rates = (0, mpmath.inf)
        for i, input_rate in enumerate(input_rates):
            dfs_args.input_rate = input_rate
            min_rate = prev_seqs[i].distillation_rate(memory, input_rate) if prev_seqs[i] else 0.0
            seq = dfs_code_sequence(dfs_args, init_seq, min_rate, print_progress=False)
            seqs[i] = seq
        
        with static_args.lock:
            if static_args.shared["M"] < memory:
                static_args.shared["M"] = memory
                static_args.shared["seqs"] = tuple(seqs)

        # Collect results
        results = [{
            "memory": int(memory),
            "input_rate": str(input_rate),
            "sequence": seq.serialise() if seq else None
        } for seq, input_rate in zip(seqs, input_rates)]
        
        return results


def main():
    # Static job arguments
    # in_error, targ_error = mpmath.mpf("1.25e-2"), mpmath.mpf("1e-12")  # p_bell = 1%, p_targ = 1e-12
    in_error, targ_error = mpmath.mpf("1.25e-2"), mpmath.mpf("1e-6")  # p_bell = 1%, p_targ = 1e-6    
    # in_error, targ_error = mpmath.mpf("5.2e-2"), mpmath.mpf("1e-6")  # p_bell = 5%, p_targ = 1e-6
    
    no_growing = False
    # code_size_step_size = 5
    code_size_step_size = 1

    # Parameter space
    memory_arr = np.unique(np.logspace(3, 5, 1000).astype(int))
    min_memory = 1250 # Already know no solutions below this point
    max_memory = 20_000 # Don't need data above this point
    memory_arr = memory_arr[memory_arr > min_memory]
    memory_arr = memory_arr[memory_arr < max_memory]

    # File to write to
    data_file_rel_path = 'data/test.dat'
    
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
        iterator = pool.imap_unordered(job_parallel, memory_arr, chunksize=chunksize)
        for results in tqdm(iterator, total=len(memory_arr)):
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