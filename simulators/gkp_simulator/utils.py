import numpy as np
from itertools import product
from functools import reduce
from ..cv_simulator.mps import (MPS, tensor_svd)
from ..cv_simulator.utils import *
from ..dv_simulator import numpy_quantum as npq

PI = np.pi
SQPI = np.sqrt(np.pi)

def eps2db(epsilon: float) -> float:
    return -10.0 * np.log10(2.0 * np.tanh(epsilon / 2.0))

def db2eps(db_squeezing: float) -> float:
    return 2.0 * np.atanh(np.float_power(10.0, -db_squeezing / 10.0) / 2.0)

def decomp_result(s: float) -> tuple[int, float]:
    # Compute n, r such that s = (n + r)√π
    n = np.round(s / np.sqrt(np.pi)).astype(int)
    r = s / np.sqrt(np.pi) - n
    return n, r

def format_result(s: float, dec: int = 4) -> str:
    n, r = decomp_result(s * 2**0.5)
    return f"({n}{r:+.{dec}f})√π"

def cv2dv_information(s: float) -> bool:
    # Compute the parity of the closest multiple of SQPI
    return np.round(s/SQPI) % 2 == 1

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

def full_logical_density_mps(mps: MPS, normalised: bool=False) -> np.ndarray:
    # Logical density matrix as defined in appendix D in M. H. Shaw et al. "Logical Gates and Read-Out of Superconducting 
    # Gottesman-Kitaev-Preskill Qubits." Apr. 5, 2024. arXiv: 2403.02396[quant-ph].

    qs = mps.domain
    dq = (qs[-1] - qs[0]) / len(qs)
    q_thingy = qs[:, np.newaxis] - qs[np.newaxis, :]

    # construct Pauli measurement operators (should ideally be precomputed in the future)
    Im = np.identity(len(qs))
    Xm = np.zeros((len(qs), len(qs)))
    Zm = np.zeros((len(qs), len(qs)))
    max_m = int((qs[-1] - qs[0]) / SQPI) + 1
    for n, m in enumerate(range(1, max_m, 2)):
        # m = 2n+1
        coeff = (-1)**(n%2) * 2 / (m * PI)
        
        # we use sinc interpolation for displacement
        T1 = np.sinc((q_thingy - m*SQPI) / dq)
        T2 = np.sinc((q_thingy + m*SQPI) / dq)
        Xm += coeff * (T1 + T2)

        # we combine linear phases into one term
        # T1 = np.diag(np.exp(1j * SQPI * (+m) * qs))
        # T2 = np.diag(np.exp(1j * SQPI * (-m) * qs))
        T1_plus_T2 = np.diag(2 * np.cos(SQPI * m * qs))
        Zm += coeff * T1_plus_T2
    Ym = 1j * Xm @ Zm

    # Pauli measurement operators
    Pms = [Im, Xm, Ym, Zm]
    # Pauli logical operators
    Ps = [
        np.array([[1, 0], [0, 1]]),    # I
        np.array([[0, 1], [1, 0]]),    # X
        np.array([[0, -1j], [1j, 0]]), # Y
        np.array([[1, 0], [0, -1]]),   # Z
    ]
    
    # Construct density matrix representation rho
    N = len(mps)
    rho = np.zeros((2**N, 2**N), dtype=complex)
    for index in product(*[[0, 1, 2, 3],]*N):
        coeff = np.ones((1, 1))
        for i, m in zip(index, mps):
            coeff = np.einsum("ab,aci,bdj,dc -> ij", coeff, m, np.conj(m), Pms[i], optimize=True)
        coeff = coeff[0, 0] * (dq/2)**N
        
        logical_pauli = reduce(np.kron, [Ps[i] for i in index], 1)
        rho += coeff * logical_pauli
    
    if normalised:
        rho /= np.trace(rho)
    
    return rho

def full_logical_density(qs: np.ndarray, state: np.ndarray) -> np.ndarray:
    # Lazy implementation
    tensors = []
    state = np.reshape(state, (1, *state.shape, 1))
    for _ in range(state.ndim - 3):
        m, state = tensor_svd(state, (0, 1), tuple(range(2, state.ndim)))
        tensors.append(m)
    tensors.append(state)
    
    return full_logical_density_mps(MPS(qs, tensors))