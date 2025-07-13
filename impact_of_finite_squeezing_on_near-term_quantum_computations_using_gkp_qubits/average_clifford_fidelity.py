import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from tqdm import tqdm
from itertools import product as iprod
from multiprocessing import Pool

from simulators.dv_simulator import numpy_quantum as npq
from simulators.cv_simulator.mps import MPS
from simulators.cv_simulator.states import eval_gkp_state, State
from simulators.gkp_simulator.utils import full_logical_density_mps, db2eps


def encode_ket(qs: np.ndarray, epsilon: float, ket: np.ndarray) -> MPS:
    ket = npq.normalise(ket)
    N = npq.num_qubits(ket)

    if N == 1:
        state = eval_gkp_state(qs, epsilon, ket)
        return MPS(qs, [np.reshape(state, (1, -1, 1))])
    
    basis_states: list[list[State]] = []
    coeffs: list[float] = []
    for i, coeff in enumerate(ket):
        if np.isclose(np.abs(coeff), 0):
            continue
        binary = "{0:0{1}b}".format(i, N)
        state = [State.GKP_ZERO if digit=="0" else State.GKP_ONE for digit in binary]
        basis_states.append(state)
        coeffs.append(coeff)
    
    M = len(basis_states)
    d = len(qs)
    tensors = []
    
    # Leftmost tensor
    tensor = np.zeros((1, d, M), dtype=complex)
    for j in range(M):
        state = basis_states[j][0]
        tensor[0, :, j] = state.eval(qs, epsilon) * coeffs[j]
    tensors.append(tensor)

    # Middle tensors
    for i in range(1, N-1):
        tensor = np.zeros((M, d, M), dtype=complex)
        for j in range(M):
            state = basis_states[j][i]
            tensor[j, :, j] = state.eval(qs, epsilon)
        tensors.append(tensor)
    
    # Rightmost tensor
    tensor = np.zeros((M, d, 1), dtype=complex)
    for j in range(M):
        state = basis_states[j][-1]
        tensor[j, :, 0] = state.eval(qs, epsilon)
    tensors.append(tensor)

    return MPS(qs, tensors)

def compute_paulis():
    # Compute 16 phase-free two-qubit Paulis (modulo ±1, ±i)
    paulis = []
    for u1, v1, u2, v2 in iprod([0, 1], repeat=4):
        P1 = (npq.X if u1 else npq.IDTY) @ (npq.Z if v1 else npq.IDTY)
        P2 = (npq.X if u2 else npq.IDTY) @ (npq.Z if v2 else npq.IDTY)
        paulis.append(np.kron(P1, P2))
    return paulis

paulis = compute_paulis()

def pauli_symplectic_label(P):
    # Determine what Pauli (up to phase) and return its symplectic representation
    for idx, (u1, v1, u2, v2) in enumerate(iprod([0,1], repeat=4)):
        candidate = paulis[idx]
        # remove phase
        i, j = np.argwhere(np.abs(candidate) > 1e-8)[0]
        c = P[i, j] / candidate[i, j]
        if np.allclose(P, candidate * c):
            return (u1, u2, v1, v2)
    raise ValueError("Not a Pauli operator!")

def symplectic_rep(U):
    basis = [
        npq.tensor(npq.X, npq.IDTY), # X1
        npq.tensor(npq.IDTY, npq.X), # X2
        npq.tensor(npq.Z, npq.IDTY), # Z1
        npq.tensor(npq.IDTY, npq.Z), # Z2
    ]
    M = np.zeros((4, 4), dtype=int)
    for col, P in enumerate(basis):
        P_image = U @ P @ npq.dagger(U)
        M[:, col] = pauli_symplectic_label(P_image)
    return M % 2

def compute_cliffords():
    # The 6 generators of the Clifford group that we consider
    generators = [
        npq.tensor(npq.H, npq.IDTY),
        npq.tensor(npq.IDTY, npq.H),
        npq.tensor(npq.P, npq.IDTY),
        npq.tensor(npq.IDTY, npq.P),
        npq.CX,
        npq.permute_tensor_product(npq.CX, [1, 0]),
        npq.SWAP,
    ]
    # Symplectic representation of generators
    generators_sympl = [(symplectic_rep(g), g) for g in generators]

    # Hashing and unhashing function for symplectic matrices
    def hash(arr: np.ndarray):
        return tuple(map(tuple, arr))

    def unhash(hash):
        return np.array(hash)
    
    # BFS compute all unique symplectic representations (equivalence classes) and store one unitary representative of each 
    # (Also store minimum distance. The max min distance is the min depth needed to sample from all Cliffords).
    idty = np.eye(4, dtype=int)
    hashmap = {hash(idty): (idty.astype(complex), 0)}
    queue = [idty]
    while queue:
        S = queue.pop(0)
        U, d = hashmap[hash(S)]
        d_new = d + 1
        for Sg, Ug in generators_sympl:
            S_new = (Sg @ S) % 2
            key = hash(S_new)
            if key not in hashmap:
                hashmap[key] = (Ug @ U, d_new)
                queue.append(S_new)
            elif hashmap[key][1] > d_new:
                hashmap[key] = (Ug @ U, d_new)

    # All the unitary representatives
    cliffords_mod_Paulis = [unitary for unitary, _ in hashmap.values()]
    print("Enumerated symplectic reps:", len(cliffords_mod_Paulis)) # should be 720
    print("Full coverage depth (Cayley graph diameter):", max(d for _, d in hashmap.values())) # is 7 for the current generator set

    # # Compute all the 2-qubit Cliffords by cross product of the 16 Paulis and each equivalence class
    # cliffords = []
    # for U_M in cliffords_mod_Paulis:
    #     for P in paulis:
    #         cliffords.append(P @ U_M)
    # print("Total Cliffords generated:", len(cliffords)) # should be 11520

    return cliffords_mod_Paulis

cliffords_mod_Paulis = compute_cliffords()

def test(db: float) -> float:
    res = 0
    ket = np.array([1, 0, 0, 0])
    for c in cliffords_mod_Paulis:
        for p in paulis:
            res += abs(ket @ p @ c @ ket)**2
    res /= len(cliffords_mod_Paulis) * len(paulis)
    print("Average survival rate:", res) # should be 1/4
    
    qs = np.linspace(-20, 20, 1000)
    print("dB:", db)  
    print("Encoding fidelities:")  
    for clifford in cliffords_mod_Paulis:
        ket = clifford @ np.array([1, 0, 0, 0])
        mps = encode_ket(qs, db2eps(db), ket)
        rho = full_logical_density_mps(mps, True)
        print(npq.fidelity(rho, ket))
   
def job(arg):
    qs = np.linspace(-20, 20, 1000)
    db, clifford_idx = arg
    
    ket = np.array([1, 0, 0, 0])  # |00>
    ket = cliffords_mod_Paulis[clifford_idx] @ ket
    mps = encode_ket(qs, db2eps(db), ket)
    rho = full_logical_density_mps(mps, True)
    
    fidelities = []
    for p in paulis:
        fidelities.append(npq.fidelity(p @ ket, rho))

    result = {
        "db": db,
        "clifford_index": clifford_idx,
        "fidelities": fidelities
    }
    return result

def main():
    dbs = np.linspace(5, 15, 13)[:2]

    data_file = 'gkp_cliff.dat'
    
    num_jobs = 3
    chunksize = 10
    write_time = 50
    
    # Setup writable
    data_file = SCRIPT_DIR + "/data/" + data_file
    if os.path.exists(data_file):
        raise FileExistsError(f"File {data_file} already exists!")
    open(data_file, 'w').close()
    
    data = []
    args = iprod(dbs, range(len(cliffords_mod_Paulis)))
    total = len(dbs) * len(cliffords_mod_Paulis)
    # Parrallelise
    with Pool(num_jobs) as pool:
        # Parallelised iterator
        iterator = pool.imap_unordered(job, args, chunksize=chunksize)
        # Add progress bar
        for result in tqdm(iterator, total=total, smoothing=0.0):
            # Record results
            data.append(result)
            # Write to disk once in a while
            if len(data) % write_time == 0:
                with open(data_file, 'w') as file:
                    file.write(json.dumps(data))
    
    # Make sure all data is written to disk
    with open(data_file, 'w') as file:
        file.write(json.dumps(data))


if __name__ == "__main__":
    main()

