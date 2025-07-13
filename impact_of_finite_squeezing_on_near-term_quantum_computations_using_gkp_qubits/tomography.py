from typing import Callable
from itertools import product as iprod
import numpy as np
from simulators.dv_simulator import numpy_quantum as npq

type Channel = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

"""
Returns function given by mapping between 2D density matrices E(rho), where E is the quantum 
channel defined by the given Krauss operators. If ket_input is True (default False) the returned 
function expects inputs that are 1D kets. If return_input is True, the function will return an
(input, output) tuple of density matrices.
"""
def quantum_channel(
    Ks: list[np.ndarray] | tuple[np.ndarray, list[np.ndarray]],
    *,
    ket_input: bool=False,
    return_input: bool=False,
    normalise: bool=False,
):
    if isinstance(Ks, tuple) and len(Ks) == 2 and isinstance(Ks[1], list):
        c1 = lambda rho: sum([d * K @ rho @ npq.dagger(K) for d, K in zip(*Ks)])
    else:
        c1 = lambda rho: sum([K @ rho @ npq.dagger(K) for K in Ks])
    
    if normalise:
        c2 = lambda rho: npq.normalise(c1(rho))
    else:
        c2 = c1
    
    if return_input:
        c3 = lambda rho: (rho, c2(rho))
    else:
        c3 = c2
    
    if ket_input:
        c4 = lambda ket: c3(npq.ket2dm(ket))
    else:
        c4 = c3
    
    return c4


def state_basis(N: int) -> list[np.ndarray]:
    ket_n = [npq.tensor(*kets) for kets in iprod(*[[npq.ZERO, npq.ONE]]*N)]
    rho_nm = [np.outer(n, m) for n, m in iprod(ket_n, ket_n)]
    return rho_nm


def pure_state_basis_kets(N: int) -> list[np.ndarray]:
    ket_n = [npq.tensor(*kets) for kets in iprod(*[[npq.ZERO, npq.ONE]]*N)]

    basis = ket_n.copy()
    for i, n in enumerate(ket_n):
        for m in ket_n[i+1:]:
            basis.append((n + m) * 2**-0.5)
            basis.append((n + 1j*m) * 2**-0.5)
    
    return basis


def operator_basis(N: int) -> list[np.ndarray]:
    paulis = [
        npq.IDTY / np.sqrt(2),
        npq.X / np.sqrt(2),
        npq.Y / np.sqrt(2),
        npq.Z / np.sqrt(2),
    ]

    basis = [npq.tensor(*opers) for opers in iprod(*[paulis]*N)]
    return basis

"""
Compute process matrix from the samples of a quantum process. inputs and outputs should be 
lists of density matrices. Given an oversampled mapping, a least squares best fit will be 
used. Raises an error given an undersampled mapping.
"""
def process_matrix(inputs: list[np.ndarray], outputs: list[np.ndarray]) -> np.ndarray:
    if len(inputs) != len(outputs):
        raise ValueError("Inconsistent number of inputs to outputs.")
    
    # We get the superoperator representation by using .flatten() only because
    # state_basis(N) is defined to be consistent with this convention.
    A = np.array([rho.flatten() for rho in inputs ]).T
    B = np.array([rho.flatten() for rho in outputs]).T

    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    # Remove small singular values using same cutoff as numpy
    cutoff = max(A.shape) * np.finfo(A.dtype).eps * max(S)
    nonzero = S > cutoff

    # Rank check
    rank = np.sum(S > cutoff)
    if rank < A.shape[1]:
        raise ValueError(f"Insufficiently sampled input set.")

    # Construct A's pseudoinverse
    S_inv = np.zeros_like(S)
    S_inv[nonzero] = 1.0 / S[nonzero]
    S[~nonzero] = 0
    A_pinv = Vh.conj().T @ np.diag(S_inv) @ U.conj().T

    # Least-squares best fit
    M = B @ A_pinv
    return M


def lambda_inv(N: int) -> np.ndarray:
    d = 4**N
    
    Lambda = np.zeros((d,)*4, dtype=complex)  # ijmn
    for i, e in enumerate(state_basis(N)):
        for m, E1 in enumerate(operator_basis(N)):
            for n, E2 in enumerate(operator_basis(N)):
                temp = E1 @ e @ E2  # npq.dagger(E2)  # We use Hermitian Pauli basis
                Lambda[i, :, m, n] = temp.flatten()

    Lambda_inv = np.linalg.pinv(np.reshape(Lambda, (d**2, d**2)))
    Lambda_inv = np.reshape(Lambda_inv, (d,)*4)  # mnij
    return Lambda_inv


def chi_matrix(process_matrix: np.ndarray, N: int, *, strict: bool=False) -> np.ndarray:    
    chi = np.einsum("mnij, ij -> mn", lambda_inv(N), process_matrix)
    
    if strict:
        # CP test
        if not np.allclose(chi, chi.conj().T):
            # Strictly speaking we should also check nonnegative real eigenvalues.
            raise ValueError("Chi matrix not trace preserving (TP)")
        
        # TP test
        basis = operator_basis(N)
        test = 0
        for n, Pn in enumerate(basis):
            for m, Pm in enumerate(basis):
                test += chi[n, m] * Pm @ Pn
        if not np.allclose(test, np.identity(test.shape[0])):
            raise ValueError("Chi matrix not trace preserving (TP)")

    return chi


def krauss_operators(chi: np.ndarray, N: int) -> tuple[np.ndarray, list[np.ndarray]]:
    # Assume Hermitian chi matrix since this is true for quantum channels
    # due to quantum channels preserving Hermitian operators.
    D, U = np.linalg.eigh(chi)
    
    Ks = []
    # Collumns of U are the eigenvectors of chi
    for eigvec in U.T:
        K = sum([oper * val for oper, val in zip(operator_basis(N), eigvec)])
        Ks.append(K)
        
    return D, Ks


"""
Evaluate process at inputs that are likely to be sufficient for optaining a process matrix. 
This may still fail if the inputs returned by the process are badly behaved.
"""
def eval_process(process: Channel, N: int, ket_input: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs = pure_state_basis_kets(N)
    outputs = []
    for i, ket in enumerate(inputs):
        input, output = process(ket) if ket_input else process(npq.ket2dm(ket))
        inputs[i] = input
        outputs.append(output)
    
    return inputs, outputs


"""
Quantum process tomography computes the Krauss operators {K_i} of the given process such that
    process(rho) = sum_i K_i rho K_i^dagger
Tomography will fail if given process does not represent a CPTP map.

`process` should be a mapping between two N-qubit spaces. The output should be a tuple of two
2D density matrices, (input, output), where the second entry is the resulting density matrix of 
applying the process to the input density matrix in the first entry.
`ket_input` indicates whether the process expects 1D kets or 2D density matrices as the input.
`normalised`. If True, the normalised Krauss operators are returned together with their 
corresponding eigenvalues.
If `full_output` is False, only the Krauss operators with eigenvalues > `cutoff` are returned.
"""
def process_tomography(
    process: Channel, 
    N: int, 
    *, 
    ket_input: bool=True,
    normalised: bool=False,
    full_output: bool=False,
    strict: bool=False,
    cutoff: float=1e-12,
) -> list[np.ndarray] | tuple[np.ndarray, list[np.ndarray]]:

    M = process_matrix(*eval_process(process, N, ket_input))
    chi = chi_matrix(M, N, strict=strict)
    
    if not np.allclose(chi, npq.dagger(chi)):
        raise ValueError("Process is not a CPTP map!")
    
    D, Ks = krauss_operators(chi, N)
    
    # Remove neglectible operators, and add eigvals to Kraus operators
    if not full_output:
        filter = D > cutoff
        D = D[filter]
        Ks = [K for K, f in zip(Ks, filter) if f]
    
    if normalised:
        return D, Ks
    else:
        return [np.sqrt(d) * K for d, K in zip(D, Ks)]