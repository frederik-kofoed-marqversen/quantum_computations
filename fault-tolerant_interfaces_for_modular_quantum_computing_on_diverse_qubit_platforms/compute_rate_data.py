import json
import numpy as np
from collections import defaultdict
from typing import Callable
from dataclasses import dataclass
from bisect import bisect_right
from sequence_class import LogicalDistillationSequence as DistillationSequence
from physical_distillation import PhysicalDistillationRateExtrapolator
from tqdm import tqdm
from utils import *


def load_sequences(path: str) -> dict[int, list[DistillationSequence]]:
    with open(path, 'r') as file:
        data = json.load(file)

    # Load in all sequences in order of the memory at which they were found
    all_sequences = defaultdict(list)
    for entry in data:
        M = entry["memory"]
        seq = entry["sequence"]
        if seq is None:
            continue
        seq = DistillationSequence.deserialise(seq)
        all_sequences[M].append(seq)
    return all_sequences


# Distillation with and without growing
def compute_distillation_data(path: str, r_rel: np.ndarray, Ms: np.ndarray) -> np.ndarray:
    all_sequences = load_sequences(path)
    loaded_Ms = sorted(all_sequences.keys())
    all_sequences: list[list[DistillationSequence]] = [all_sequences[M] for M in loaded_Ms]
    
    if Ms[-1] > loaded_Ms[-1] + 1000:
        raise ValueError("Insufficient data. Distillation rates will be suboptimal!")

    # Filter out duplicates:
    hashset = set()
    filtered_sequences: list[list[DistillationSequence]] = [[]]*len(all_sequences)
    for i, seqs in enumerate(all_sequences):
        seqs = [seq for seq in seqs if seq.serialise() not in hashset]
        hashset = hashset.union({seq.serialise() for seq in seqs})
        filtered_sequences[i] = seqs
    print("Total number of sequences:", sum(len(seqs) for seqs in all_sequences))
    print("Number of unique sequences:", len(hashset))
    
    # Compute optimal rate for each memory parameter
    rate = np.zeros((len(r_rel), len(Ms)), dtype=object)
    for j, M in enumerate(Ms):
        index = bisect_right(loaded_Ms, M)
        if index == 0:
            continue
        
        rate_M = np.zeros(len(r_rel), dtype=object)
        seqs = sum(filtered_sequences[:index], [])
        for seq in seqs:
            E = seq.encoding_rate
            C = seq.input_rate_cap(M)
            rate_S = E * np.minimum(r_rel, C)
            rate_M = np.maximum(rate_M, rate_S)
        
        rate[:, j] = rate_M
    
    # NOTE: The `rate` is in units of physical gate rate.
    return rate


@dataclass
class RateArgs:
    r_rel: np.ndarray
    Ms: np.ndarray
    p_target: float
    p_physical: float
    p_bell: float
    p_idle: float
    sequence_file: str


@dataclass
class RateData:
    Z: np.ndarray
    ids: np.ndarray
    rs: list[np.ndarray]
    rate_labels: list[str]
    memory_unit: int
    Ms: np.ndarray
    r_rel: np.ndarray


def compute_rate_data(args: RateArgs, *, do_LS=True, do_T=True, do_D=True) -> RateData:
    r_rel, Ms, p_target, p_physical = args.r_rel, args.Ms, args.p_target, args.p_physical
    shape = (len(r_rel), len(Ms))

    # Helper function

    ideling_channel = DepolarisationChannel(args.p_idle)

    def surface_code_error_rate(L: int, idle_time: Callable) -> float:
        p_seam = ideling_channel.apply(args.p_bell, idle_time(L), True)
        p = logical_error_rate_bulk_seam(L, p_physical, p_seam)
        return p

    def surface_code_size(idle_time: Callable) -> int:
        L, p = find_code_size(surface_code_error_rate, p_target, args=(idle_time,), stepsize=10, always_return=True)
        if p > p_target:
            return None
        return L

    L_T: Callable[[float], int] = lambda r_bell: surface_code_size(lambda L: L**2 / r_bell)
    L_LS: Callable[[float], int] = lambda r_bell: surface_code_size(lambda L: L / r_bell)
    L_D: int = surface_code_size_bulk_seam(p_physical, 0, p_target)
    
    # Transversal
    rs_T = np.full(shape, 0.0, dtype=object)
    if do_T:
        print("Computing transversal gate rates.")
        L_Ts = [L_T(r) for r in r_rel]
        for i, j in np.ndindex(shape):
            r, L, M = r_rel[i], L_Ts[i], Ms[j]
            rs_T[i, j] = transversal_gate_rate(L, 1, r, M) if L else 0

    # Lattice surgery
    rs_LS = np.full(shape, 0.0, dtype=object)
    if do_LS:
        print("Computing lattice surgery rates.")
        L_LSs = [L_LS(r) for r in r_rel]
        for i, j in np.ndindex(shape):
            r, L, M = r_rel[i], L_LSs[i], Ms[j]
            rs_LS[i, j] = lattice_surgery_gate_rate(L, 1, r, M) if L else 0

    # Logical distillation
    rs_D = np.full(shape, 0.0, dtype=object)
    if do_D:
        print("Computing distillation rate")
        rs_D = compute_distillation_data(args.sequence_file, r_rel, Ms) if args.sequence_file else np.zeros(shape)

    # Compute maximum rate
    rate_labels = ["Transversal", "Lattice surgery", "Distillation"]
    rs = [rs_T, rs_LS, rs_D]
    # NOTE: Rates are in units of physical gate rate. To get units of logical gate rate multiply by 5!
    rs = [r * 5 for r in rs]

    Z = np.stack(rs)
    ids = np.argmax(Z, axis=0)
    Z = np.max(Z, axis=0)
    ids[Z==0] = -1

    return RateData(Z, ids, rs, rate_labels, L_D, Ms, r_rel)


def add_physical_distillation(r_rel: np.ndarray, Ms: np.ndarray, second_stage_data: RateData) -> tuple[np.ndarray, np.ndarray]:
    Z_2nd, ids_2nd = second_stage_data.Z, second_stage_data.ids
    Ms_2nd, r_rel_2nd = second_stage_data.Ms, second_stage_data.r_rel

    pd_rate = PhysicalDistillationRateExtrapolator("./data/physical_distillation.dat", max_mem=Ms[-1])
    dM = int(np.mean(np.diff(Ms)))
    Ms_ext = list(range(0, Ms[0], dM)) + Ms.tolist()

    print("Computing transformed image. This may take a while...")
    shape = (len(r_rel), len(Ms))
    Z2 = np.zeros(shape, dtype=object)
    ids2 = np.full(shape, -1)
    for i, r in tqdm(enumerate(r_rel), total=len(r_rel)):
        r_stars = [pd_rate.eval(r, M) for M in Ms_ext]

        for j, M_tot in enumerate(Ms):
            # Record values with physical distillation
            r_list = []
            id_list = []
            for M, r_star in zip(Ms_ext, r_stars):
                M_star = M_tot - M
                if M_star < 0:
                    break
                x_idx = bisect_right(r_rel_2nd, r_star) - 1
                y_idx = bisect_right(Ms_2nd, M_star) - 1
                if x_idx < 0 or y_idx < 0:
                    # Transformed value fall outside of known region
                    continue

                r_list.append(Z_2nd[x_idx, y_idx])
                id_list.append(ids_2nd[x_idx, y_idx])

            # Record only results from optimal amount of memory allocated to physical distillation
            if len(r_list) == 0:
                continue
            index = np.argmax(r_list)
            Z2[i, j] = r_list[index]
            ids2[i, j] = id_list[index]
    
    return Z2, ids2