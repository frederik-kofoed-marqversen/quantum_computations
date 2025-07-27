import os
import itertools as itt
import pandas as pd
import rtree as rt
from collections import defaultdict
from sequence_class import LogicalDistillationSequence, GrowStage, ClassicalStage, QuantumStage
from mpmath import inf, isinf
# from sequence_simulation import Simulator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import logging
logger = logging.getLogger(__name__)


class DFSArgs:
    def __init__(
        self,
        physical_error_rate: float,
        memory: int,
        target_error: float,
        target_size: int,
        rel_input_rate: float,
        *,
        max_seq_len: int = inf,
        code_sizes: list[int] = None
    ):
        self.p_local = physical_error_rate
        self.max_seq_len = max_seq_len
        self.memory = memory
        self.target_error = target_error
        self.target_size = target_size
        # input rate in units of local gate rate!
        self.input_rate = rel_input_rate

        self.cl_codes = None
        self.q_codes = None
        self.code_sizes = code_sizes
    
    def shallow_copy(self) -> 'DFSArgs':
        copy = DFSArgs(
            self.p_local,
            self.memory,
            self.target_error,
            self.target_size,
            self.input_rate,
            max_seq_len=self.max_seq_len,
        )
        copy.cl_codes = self.cl_codes
        copy.q_codes = self.q_codes
        copy.code_sizes = self.code_sizes
        return copy
    
    def init_codes(self, max_rep_code: int=inf, max_quantum_code: int=inf) -> None:
        # Read in error correction codes
        data_frame = pd.read_excel(f'{SCRIPT_DIR}/ConstantRateDistillation/CodesTable_All_Expanded.xlsx')
        
        # for classical codes - keep only repetition codes.
        # Also, we have observed that large classical codes are not relevant
        # so keep only codes [n, k, d] with n <= max_rep_code.
        max_rep_code = min(12, max_rep_code)
        mask = (data_frame['CodeType'] == 'Classical') & ((data_frame['n'] != data_frame['d']) | (data_frame['n'] > max_rep_code))
        data_frame = data_frame[~mask]
        # Keep only quantum codes [[n, k, d]] with n <= max_quantum_code
        mask = (data_frame['CodeType'] == 'Quantum') & ((data_frame['n'] > max_quantum_code) | (data_frame['n'] == 1))
        data_frame = data_frame[~mask]
        
        # Separate code types
        q_codes = data_frame[data_frame['CodeType'] == 'Quantum'].to_numpy()
        q_codes = sorted(q_codes, key=lambda c: (c[0], -c[1], -c[2]))
        self.q_codes = q_codes
        self.cl_codes = data_frame[data_frame['CodeType'] == 'Classical'].to_numpy()
    
    def init_code_sizes(self, L_init):
        # Filter non-usable sizes
        code_sizes = self.code_sizes if self.code_sizes is not None else list(range(self.target_size))
        code_sizes = [L for L in code_sizes if L < self.target_size and L > L_init]
        if self.target_size > L_init:
            code_sizes.append(self.target_size)
        self.code_sizes = code_sizes

"""
Each sequence is described by a parameter tuple (L, p1, p2,...). The objective function 
(distillation rate) after adding sequence of steps S is r(L, pi, S). One can prove that
r is monotonously decreasing in each parameter pi independent of S. So for a given L, if
a sequence has all pi's larger than a previously explored sequence, it can only lead to
a smaller rate, implying that the branch extending from the sequence can be cut/pruned.
"""
class CachedPruner:
    def __init__(self, max_M: int=1e9, max_K: int=1e3):
        p = rt.index.Property(dimension=5)
        self.rtrees: defaultdict[int, rt.Index] = defaultdict(lambda : rt.Index(properties=p))
        self._max_vals = (1.0, max_K, 0.0, max_M, max_M)
        self._id_gen = itt.count()
    
    @property
    def size(self) -> int:
        return sum(rtree.get_size() for rtree in self.rtrees.values())

    def _parse_sequence(self, sequence: LogicalDistillationSequence) -> tuple[rt.Index, tuple]:
        L = sequence.L
        K, E, p = sequence.K, sequence.encoding_rate, sequence.p_out
        M, Midle = sequence.M, sequence.M_idle
        return self.rtrees[L], (p, K, -E, M, Midle)
    
    def prune(self, sequence: LogicalDistillationSequence) -> bool:
        rtree, point = self._parse_sequence(sequence)
        iterator = rtree.intersection((*point, *point))
        prune = any(True for _ in iterator)
        return prune
    
    def insert_prune_value(self, sequence: LogicalDistillationSequence) -> None:
        rtree, point = self._parse_sequence(sequence)
        bounds = (*point, *self._max_vals)
        uid = next(self._id_gen)
        rtree.insert(uid, bounds)


def _dfs_add_distillation_branches(
    args: DFSArgs,
    current: LogicalDistillationSequence,
    best: LogicalDistillationSequence,
    pruner: CachedPruner,
    print_progress: bool=False,
) -> LogicalDistillationSequence:
    prev_stage = current.stages[-1]
    cl_code_basis = prev_stage.basis if isinstance(prev_stage, ClassicalStage) else None    
    for code in itt.chain(args.cl_codes, args.q_codes):
        new = current.shallow_copy()
        new_args = args.shallow_copy()
        if code[3] == "Quantum":
            new.add_stage(QuantumStage(code[:3], new.L, new.p_L, args.p_local))
            # Ignore classical codes after quantum ones
            new_args.cl_codes = []
        elif code[4] == cl_code_basis:
            # Never do two consecutive classical codes on same axis
            continue
        else:
            new.add_stage(ClassicalStage(code[:3], code[4], new.L, new.p_L, args.p_local))
        
        # Skip if error is worse
        if new.p_out > current.p_out:
            continue
        
        # Compute best sequence from new
        best = _dfs_recursive(new_args, new, best, pruner, print_progress)
    
    return best


def _dfs_add_growing_branches(
    args: DFSArgs,
    current: LogicalDistillationSequence,
    best: LogicalDistillationSequence,
    pruner: CachedPruner,
    print_progress: bool=False,
) -> LogicalDistillationSequence:    
    for i, L in enumerate(reversed(args.code_sizes)):
        # Add grow stage
        new = current.shallow_copy()
        new.add_stage(GrowStage(L, new.L, new.p_L, args.p_local))
        
        # Never grow backwards
        new_args = args.shallow_copy()
        new_args.code_sizes = args.code_sizes[len(args.code_sizes)-i:]
        
        # Compute best sequence from new
        best = _dfs_recursive(new_args, new, best, pruner, print_progress)
    
    return best

prune_counter = itt.count()

def _dfs_recursive(
    args: DFSArgs,
    current: LogicalDistillationSequence,
    best: LogicalDistillationSequence,
    pruner: CachedPruner,
    print_progress: bool=False,
) -> LogicalDistillationSequence:
    # Check cached pruning
    if pruner.prune(current):
        if print_progress:
            global prune_counter
            count = next(prune_counter)
            if count % 10_000 == 0:
                print(f"Pruner \t Size: {pruner.size}. Count: {count}")
        return best
    
    # Elevate current sequence to potential solution
    test = current.shallow_copy()
    if test.L < args.target_size:
        test.add_stage(GrowStage(args.target_size, test.L, test.p_L, args.p_local))
    test_rate = test.distillation_rate(args.memory, args.input_rate)
    if test_rate == 0.0:
        return best
    if test_rate <= best._distillation_rate:
        return best
    if test.p_out < args.target_error:
        # Double check rate with monte carlo simulation
        # sim = Simulator(args.memory, args.input_rate, test)
        # test_rate = sim.estimate_rate()
        # if test_rate < best._distillation_rate:
        #     return best
        
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
    
    # Add distillation stages
    # We never distil below error rate of encoding
    if current.p_out > current.p_L:
        best = _dfs_add_distillation_branches(args, current, best, pruner, print_progress)
    # Add grow stages
    # We never grow twice in a row
    if not isinstance(current.stages[-1], GrowStage):
        best = _dfs_add_growing_branches(args, current, best, pruner, print_progress)
    
    # Record that this branch has been explored
    pruner.insert_prune_value(current)

    # Return the best sequence found
    return best

def dfs_code_sequence(args: DFSArgs, init: LogicalDistillationSequence, min_rate: float=0.0, print_progress: bool=False) -> LogicalDistillationSequence | None:
    # Will only search sequences with rates above `min_rate`. If `print_progress` is true, will
    # print the optimal sequence every time a new optimum is found.

    args.target_size = max(args.target_size, init.L)
    
    # Setup
    args.init_codes(2)  # Have observed classical codes n>2 are never relevant
    args.init_code_sizes(int(init.L))
    best = LogicalDistillationSequence.__new__(LogicalDistillationSequence)
    best._distillation_rate = min_rate
    pruner = CachedPruner()

    if min_rate == 0.0 and isinf(args.memory) and isinf(args.max_seq_len):
        logger.warning("Sequence optimisation without constraints may never finish!")
    if isinf(args.memory) and len(args.code_sizes) > 0:
        logger.warning("Sequence optimisation without memory constraint and code growing may never finish!")

    # Reset counter
    global prune_counter
    prune_counter = itt.count()
    next(prune_counter)
    # Do recursive DFS
    best = _dfs_recursive(args, init, best, pruner, print_progress)
    if len(best.__dict__) > 1:
        if print_progress:
            print("Best sequence:")
            print(best)
            print(f"Distillation rate: {float(best._distillation_rate):.3e}")
            print()
        return best
    else:
        if print_progress:
            print("No valid sequence exists!")
        return None