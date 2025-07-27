import json
import numpy as np
from mpmath import mpf, isinf
from itertools import chain
from sequence_class import Stage, InitStage, ClassicalStage, QuantumStage, scalar_error
from sequence_optimisation import DFSArgs
from utils import DepolarisationChannel, find_root_bisection
from bisect import bisect_right

import logging
logger = logging.getLogger(__name__)
    

class PhysicalDistillationSequence:
    def __init__(self, init_stage: InitStage):
        self.stages: list[Stage] = [init_stage]
        self.min_memory_req: int = 0
        self.K = 1

    def __str__(self):
        lines = ["Distillation stages:"]
        for stage in self.stages:
            line = (
                f"{str(stage):<15}: "
                f"L={stage.L}, "
                f"p_L={float(stage.p_L):.3e}, "
            )
            lines.append(line)
        
        summary = f"Summary: memory requirement={self.min_memory_req}, "
        lines.append(summary)

        return "\n".join(lines)

    def serialise(self) -> str:
        return json.dumps([stage.serialise() for stage in self.stages])

    @staticmethod
    def deserialise(data_str: str) -> 'PhysicalDistillationSequence':
        serialised_stage_strs = json.loads(data_str)
        stages = [Stage.from_serialised(s) for s in serialised_stage_strs]
        seq = PhysicalDistillationSequence.__new__(PhysicalDistillationSequence)
        for stage_data in stages:
            stage = Stage.from_serialised(stage_data)
            seq.add_stage(stage)
        return seq

    def add_stage(self, stage: Stage):
        n, k = stage.n, stage.k
        K = self.K
        size = stage.qubit_size
        dM = (size - self.stages[-1].qubit_size) * n * K
        
        self.stages.append(stage)
        self.min_memory_req = max(n*K*size, (n-1)*K*size + self.min_memory_req + dM)
        self.K *= k

    def shallow_copy(self) -> 'PhysicalDistillationSequence':
        copy = PhysicalDistillationSequence.__new__(PhysicalDistillationSequence)
        copy.stages = self.stages.copy()
        copy.min_memory_req = self.min_memory_req
        copy.K = self.K
        return copy
    
    def eval_non_constrained_sequence(self, input_rate: float, *, idleing: DepolarisationChannel=None, local_gate_rate: float = 1.0) -> tuple[float, float, float]:
        M = 0
        K = 1
        E = 1
        p_out = self.stages[0].error

        for stage in self.stages[1:]:
            n, k = stage.n, stage.k
            size = stage.qubit_size
            T = stage.get_physical_depth() / local_gate_rate
            r_in = input_rate * E / (n * K)
            p_in = idleing.apply(p_out, 1/r_in) if idleing else p_out
            p_out, p_fail = stage.compute_error_metrics(p_in)

            M += size * K * (T * E * input_rate + (n-1) / 2)
            
            E *= (1 - p_fail) * k / n
            K *= k
        
        return scalar_error(p_out), M, E
    
    def eval_constrained_sequence(self, max_input_rate: float, allocated_memory: int, *, idleing: DepolarisationChannel=None, local_gate_rate: float = 1.0) -> tuple[float, float, float]:
        if self.min_memory_req > allocated_memory:
            raise ValueError("Sequence cannot be evaluated with less than minimum memory requirement.")
        
        p, M, E = self.eval_non_constrained_sequence(max_input_rate, idleing=idleing, local_gate_rate=local_gate_rate)
        if M <= allocated_memory:
            return max_input_rate, p, E
        
        fun = lambda r: allocated_memory - self.eval_non_constrained_sequence(r, idleing=idleing, local_gate_rate=local_gate_rate)[1]
        input_rate = find_root_bisection(fun, mpf("1e-6"), min(mpf("1e10"), max_input_rate))

        p, M, E = self.eval_non_constrained_sequence(input_rate, idleing=idleing, local_gate_rate=local_gate_rate)
        return input_rate, p, E


# The following DFS is just a hacked versions of the ones used for logical distillation.
# Probably should be cleaned up at some point, or a generalisation could be made.

def _dfs_recursive(
    args: DFSArgs,
    current: PhysicalDistillationSequence,
    best: PhysicalDistillationSequence,
    idleing: DepolarisationChannel,
    print_progress: bool=False,
) -> PhysicalDistillationSequence:
    # Elevate current sequence to potential solution
    test = current
    if test.min_memory_req > args.memory:
        return best
    test_rate = test._distillation_rate
    if test_rate == 0.0:
        return best
    if test_rate <= best._distillation_rate:
        return best
    if test.p_out < args.target_error:
        # New best sequence!
        if print_progress:
            print()
            print("New best sequence:")
            print(test)
            print(f"Distillation rate: {float(test_rate):.3e}")
            print()
            
        test._distillation_rate = test_rate
        return test
    if len(test.stages) >= args.max_seq_len:
        return best
    
    prev_stage = current.stages[-1]
    cl_code_basis = prev_stage.basis if isinstance(prev_stage, ClassicalStage) else None    
    for code in chain(args.cl_codes, args.q_codes):
        new = current.shallow_copy()
        new_args = args.shallow_copy()
        if code[3] == "Quantum":
            new.add_stage(QuantumStage(code[:3], 1, args.p_local, args.p_local))
            # Ignore classical codes after quantum ones
            new_args.cl_codes = []
        elif code[4] == cl_code_basis:
            # Never do two consecutive classical codes on same axis
            continue
        else:
            new.add_stage(ClassicalStage(code[:3], code[4], 1, args.p_local, args.p_local))
        
        try:
            in_rate, p_out, E = new.eval_constrained_sequence(args.input_rate, args.memory, idleing=idleing)
        except:
            logger.warning(f"Error while evaluating sequence:\n{new}\nSkipping this branch")
            continue
        new._distillation_rate = in_rate * E
        new.p_out = p_out

        # Skip if error is worse
        if new.p_out > current.p_out:
            continue
        
        # Compute best sequence from new
        best = _dfs_recursive(new_args, new, best, idleing, print_progress)
    
    # Return the best sequence found
    return best

def dfs_code_sequence(args: DFSArgs, init: PhysicalDistillationSequence, min_rate: float=0.0, print_progress: bool=False) -> PhysicalDistillationSequence | None:
    # Will only search sequences with rates above `min_rate`. If `print_progress` is true, will
    # print the optimal sequence every time a new optimum is found.
    
    # Setup
    args.init_codes(6, 6)
    p_idle = np.array([5e-6 / 25, 5e-6 / 25, 2e-5 / 25])
    idle_rate = 200  # "number of" idleing errors per physical gate
    idleing = DepolarisationChannel(p_idle, idle_rate)
    
    in_rate, p_out, E = init.eval_constrained_sequence(args.input_rate, args.memory, idleing=idleing)
    distil_rate = in_rate * E
    init._distillation_rate = distil_rate
    init.p_out = p_out

    best = PhysicalDistillationSequence.__new__(PhysicalDistillationSequence)
    best._distillation_rate = min_rate

    # Check convergence criteria
    if min_rate == 0.0 and isinf(args.memory) and isinf(args.max_seq_len):
        logger.warning("Sequence optimisation without constraints may never finish!")
    if isinf(args.memory) and len(args.code_sizes) > 0:
        logger.warning("Sequence optimisation without memory constraint and code growing may never finish!")

    # Do recursive DFS
    best = _dfs_recursive(args, init, best, idleing, print_progress)
    if len(best.__dict__) > 1:
        if print_progress:
            print("Best sequence:")
            print(best)
            print(f"Output error rate: {float(best.p_out):.3e}")
            print(f"Distillation rate: {float(best._distillation_rate):.3e}")
            print()
        return best
    else:
        if print_progress:
            print("No valid sequence exists!")
        return None


class PhysicalDistillationRateExtrapolator:
    def __init__(self, filepath: str, *, max_mem: int=None):
        with open(filepath, "r") as file:
            pd_data = json.load(file)
        self.xs = list(map(mpf, pd_data["xs"]))
        self.ys = list(map(int, pd_data["ys"]))
        self.zs = list(map(mpf, pd_data["zs"]))

        # Reducing the amount of data stored speeds up the lookup process
        # max_mem is the maximum memory that will be stored.
        if max_mem is not None:
            idx = bisect_right(self.ys, max_mem) + 1
            self.xs = self.xs[:idx]
            self.ys = self.ys[:idx]
            self.zs = self.zs[:idx]
    
    def eval(self, r, M):
        if M > self.ys[-1]:
            raise ValueError("Insufficient data for extrapolation.")
        
        x, y = r, M
        x_idx = bisect_right(self.xs, x) - 1
        y_idx = bisect_right(self.ys, y) - 1
        y_idx = max(y_idx, 0)
        
        if x >= self.xs[y_idx]:
            r = self.zs[y_idx]
        else:
            r = self.zs[x_idx]
        return r


if __name__ == "__main__":
    import os
    from multiprocessing import Pool
    from tqdm import tqdm
    import mpmath
    mpmath.mp.dps = 24

    idleing_channel = DepolarisationChannel(mpf("1e-6"))
    local_error = mpf("1e-3")
    in_error = mpf("5e-2")
    targ_error = mpf("1e-2")

    def physical_distillation(n, r_bell, M):
        seq = PhysicalDistillationSequence(InitStage(in_error, 1, local_error, local_error))
        bases = ("X", "Y")
        for i in range(n):
            seq.add_stage(ClassicalStage((2, 1, 2), bases[i%2], 1, local_error, local_error))
        
        if M < seq.min_memory_req:
            return 0, 1, 0
        input_rate, p, E = seq.eval_constrained_sequence(r_bell, M, idleing=idleing_channel)
        return input_rate, p, E

    # # Compute each individual value in the plot (slow)
    # def eval_phase_space(P, n_max=10):
    #     R = np.zeros(shape)
    #     N = np.zeros(shape)

    #     for i, j in np.ndindex(shape):
    #         if j==0 and i%5==0:
    #             print(f"Row {i} of {shape[0]}")
    #         x, y = X[i, j], Y[i, j]

    #         # Increase iterations until solution
    #         n = 0
    #         p = inf
    #         while n <= n_max:
    #             input_rate, p_new, E = physical_distillation(n, x, y)
    #             r = input_rate * E
    #             if p_new >= p:
    #                 break
    #             if r == 0.0:
    #                 break
    #             p = p_new
    #             if p < P:
    #                 N[i, j] = n
    #                 R[i, j] = r
    #                 break
    #             n += 1
    #     return R, N
    # R, N = eval_phase_space(physical_distillation, targ_error)
    # # We find that almost everywhere n=2 (only a very small window in r_rel where n=3 which only pushes the limit
    # # very slightly to the left). We also realise that we still have memory and rate caps. This can be exploited 
    # # for a faster method
    # # Also we have found that by dfs in each of the extremes, there is nowhere that quantum codes fair well

    # Faster method
    def fun(params) -> tuple[int, float, float]:
        i, y = params
        input_rate, p, E = physical_distillation(n, 1e6, y)
        if p > targ_error:
            return i, (mpf("0"), mpf("0"))
        else:
            x = input_rate
            z = input_rate * E
            return i, (x, z)

    n = 2
    ys = np.arange(0, 100_000, 1)
    zs = np.zeros_like(ys, dtype=object)
    xs = np.zeros_like(ys, dtype=object)
    # Parrallelise
    with Pool(3) as pool:
        # Parallelised iterator
        iterator = pool.imap_unordered(fun, enumerate(ys), chunksize=10)
        for i, (x, z) in tqdm(iterator, total=len(ys)):
            xs[i] = str(x)
            zs[i] = str(z)
    
    # Save data to use with PhysicalDistillationRateExtrapolator
    path = "./data/physical_distillation.dat"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data = {"xs": xs.tolist(), "ys": ys.tolist(), "zs": zs.tolist()}
    with open(SCRIPT_DIR + "/" + path, "w") as file:
        json.dump(data, file)