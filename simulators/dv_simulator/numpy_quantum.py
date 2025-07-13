import numpy as np
from functools import reduce
import builtins

ZERO, ONE = np.array([1, 0]), np.array([0, 1])
PLUS, MINUS = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
IPLUS, IMINUS = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)

IDTY = np.identity(2)
X, Y, Z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
PAULIS = [X, Y, Z]

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

CZ = np.identity(4)
CZ[3, 3] = -1

CX = np.identity(4)
CX[2:, :] = np.flip(CX[2:, :], 0)

SWAP = np.identity(4)
SWAP[1:3, :] = np.flip(SWAP[1:3, :], 0)

P = np.array([[1.0, 0.0], [0.0, 1.0j]])
T = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]])


class PauliError(ValueError):
    pass


def get_pauli_number(pauli_identifier) -> int:
    match pauli_identifier:
        case 'i' | 'I' | 0:
            return 0
        case 'x' | 'X' | 1 | [1, 0, 0]:
            return 1
        case 'y' | 'Y' | 2 | [0, 1, 0]:
            return 2
        case 'z' | 'Z' | 3 | [0, 0, 1]:
            return 3
        case '-x' | '-X' | -1 | [-1, 0, 0]:
            return -1
        case '-y' | '-Y' | -2 | [0, -1, 0]:
            return -2
        case '-z' | '-Z' | -3 | [0, 0, -1]:
            return -3
        case _:
            raise PauliError(f'"{pauli_identifier}" could not be interpreted as a Pauli operator')


def get_pauli_identifier(pauli_identifier) -> str:
    return ['-Z', '-Y', '-X', 'I', 'X', 'Y', 'Z'][get_pauli_number(pauli_identifier) + 3]


def is_pauli(case) -> bool:
    try:
        get_pauli_number(case)
        return True
    except PauliError:
        return False


def get_pauli_operator(pauli_identifier) -> np.ndarray:
    return PAULIS[get_pauli_number(pauli_identifier) - 1]


def get_pauli_states(pauli_identifier):
    return [[PLUS, MINUS], [IPLUS, IMINUS], [ZERO, ONE]][get_pauli_number(pauli_identifier) - 1]


def get_pauli_state(pauli_identifier, state_index: int) -> np.ndarray:
    return get_pauli_states(pauli_identifier)[state_index]


def basis_state(identifier, N: int) -> np.ndarray:
    match type(identifier):
        case builtins.list | builtins.tuple:
            return basis_state("".join([str(b) for b in identifier]))
        case builtins.str:
            return basis_state(int(identifier, 2), len(identifier))
        case builtins.int:
            state = np.zeros(2**N)
            state[identifier] = 1
            return state
        case ident_type:
            raise NotImplementedError(f"Could not generate basis state from identifier of type {ident_type}")


def qubit_from_polar(theta: float, phi: float):
    return np.cos(theta / 2) * ZERO + np.exp(1j * phi) * np.sin(theta / 2) * ONE


def qubit_from_axis(axis: list[float, float, float]) -> np.ndarray:
    theta = np.arccos(axis[-1] / np.sqrt(sum(a ** 2 for a in axis)))
    phi = np.arctan2(axis[1], axis[0])
    return qubit_from_polar(theta, phi)


def phase_gate(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j*theta)]]) 


def axis_rotation(theta: float, axis: list[float, float, float]) -> np.ndarray:
    return IDTY * np.cos(theta / 2) - 1j * sum(axis[i] * PAULIS[i] for i in range(3)) * np.sin(theta / 2)


def euler_rotation(theta1, theta2, theta3) -> np.ndarray:
    return axis_rotation(theta3, [1, 0, 0]) @ axis_rotation(theta2, [0, 0, 1]) @ axis_rotation(theta1, [1, 0, 0])


def ket2dm(ket: np.ndarray) -> np.ndarray:
    if len(ket.shape) != 1:
        raise TypeError('state is not a ket')
    return np.outer(ket, np.conjugate(ket))


def dm2ket(dm: np.ndarray, strict: bool=True) -> np.ndarray:
    """If strict=True an error is raised if no state properly
    represent the given density matrix. If strict=False, a best
    approximation is returned."""
    
    if not is_hermitian(dm):
        raise TypeError('input is not a density matrix')
    eigvals, eigvecs = np.linalg.eigh(dm)
    if strict and not np.allclose(eigvals[:-1], 0):
        raise TypeError('density matrix does not represent a pure state')
    return normalise(eigvecs[:, -1])


def norm(ket: np.ndarray) -> float:
    return np.linalg.norm(ket)


def normalise(state: np.ndarray) -> np.ndarray:
    if state.ndim == 1:
        return state / np.linalg.norm(state)
    elif state.ndim == 2:
        return state / np.trace(state)
    else:
        raise ValueError("State not ket nor density matrix.")


def compare_kets(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(ket2dm(normalise(a)), ket2dm(normalise(b)))


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    # If either a or b is a matrix it is assumed they are hermitian
    match (a.ndim == 1, b.ndim == 1):
        case (True, True):
            return np.abs(a.conj() @ b).real**2
        case (True, False):
            return (a.conj() @ b @ a).real
        case (False, True):
            return (b.conj() @ a @ b).real
        case (False, False):
            # This is (tr(sqrt(a @ b)))**2
            eigvals = np.linalg.eigvals(a @ b)
            eigvals = np.clip(eigvals.real, 0.0, None)
            return np.sum(np.sqrt(eigvals))**2


def purity(rho: np.ndarray) -> float:
    # Assumes rho is a hermitian density matrix
    return np.trace(rho @ rho).real


def tensor(*arrays) -> np.ndarray:
    return reduce(np.kron, arrays, 1)

def is_power_of_two(n: int) -> bool:
    # Returns True if n is a power of 2 
    # Test by bit manipulations
    return (n & (n-1) == 0) and n != 0

def is_qubit_operator(oper: np.ndarray) -> bool:
    test1 = (oper.ndim == 2)
    test2 = (oper.shape[0] == oper.shape[1])
    test3 = is_power_of_two(oper.shape[0])
    return test1 and test2 and test3


def is_qubit_state(state: np.ndarray) -> bool:
    test1 = (state.ndim == 1)
    test2 = is_power_of_two(len(state))
    return test1 and test2


def is_hermitian(oper: np.ndarray) -> bool:
    return np.allclose(dagger(oper), oper)


def expect(oper: np.ndarray, state: np.ndarray):
    if not is_qubit_operator(oper) or not is_qubit_state(state) or not oper.shape[0] == state.shape[0]:
        raise TypeError('incompatible operator and state vector')
    return np.conjugate(state) @ oper @ state


def expecth(oper: np.ndarray, state: np.ndarray):
    return expect(oper, state).real


def rand_ket(d=2) -> np.ndarray:
    return normalise(np.random.rand(d) + 1j * np.random.rand(d))


def dagger(array: np.ndarray) -> np.ndarray:
    return np.conjugate(array.T)


def _permute_tensor_product_rows(array: np.ndarray, new_ordering: list) -> np.ndarray:
    number_of_qubits = len(new_ordering)
    result = array.reshape((*[2] * number_of_qubits, -1))
    result = result.transpose([*new_ordering, number_of_qubits])
    result = result.reshape((2 ** number_of_qubits, -1))
    return result


def _permutation_inverse(perm):
    res = [0] * len(perm)
    for i in range(len(perm)):
        res[perm[i]] = i
    return res


def permute_tensor_product(array: np.ndarray, new_ordering) -> np.ndarray:
    n = array.shape[0]
    if not is_power_of_two(n):
        raise ValueError("Given array is not a qubit state nor operator")
    if set(new_ordering) != set(range(num_qubits(array))):
        raise ValueError('new_ordering must be a permutation of all qubits')
    
    new_ordering = _permutation_inverse(new_ordering)
    result = _permute_tensor_product_rows(array, new_ordering)
    if len(array.shape) == 2:
        result = _permute_tensor_product_rows(result.T, new_ordering).T
    else:
        result = result.flatten()
    return result


def expand_gate(gate: np.ndarray, N: int, targets) -> np.ndarray:
    missing_indices = [i for i in range(N) if i not in targets]
    result = tensor(gate, *[IDTY] * len(missing_indices))
    result = permute_tensor_product(result, targets + missing_indices)
    return result


def add_control(gate: np.ndarray) -> np.ndarray:
    return tensor(np.outer(ZERO, ZERO), np.identity(gate.shape[0])) + tensor(np.outer(ONE, ONE), gate)


def num_qubits(arr: np.ndarray | int) -> int:
    if isinstance(arr, int):
        return int(np.log2(arr))
    else:
        return int(np.log2(arr.shape[0]))
