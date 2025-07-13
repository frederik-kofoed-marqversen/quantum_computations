from . import numpy_quantum as npq
from .states import State
import numpy as np

REPR_DIGITS = 5

class Gate():
    def __init__(self, indices: list[int], matrix: np.ndarray | None):
        if len(set(indices)) != len(indices):
            raise ValueError("Indices must be distinct.")
        if min(indices) < 0:
            raise ValueError("Non-negative index")
        if matrix is not None:
            if matrix.ndim != 2:
                raise ValueError("Not a 2D array.")
            if any(not npq.is_power_of_two(size) for size in matrix.shape):
                raise ValueError("Given matrix is not a mapping between qubit spaces.")
            if matrix.shape[1] != 2**len(indices):
                raise ValueError("Dimensions of given matrix is not compatible with number of indices.")
        self.indices = indices
        self.matrix = matrix

    def __repr__(self):
        return type(self).__name__ + "_" + str(self.indices[0]) + "".join(f",{i}" for i in self.indices[1:])
    
    def copy(self) -> 'Gate':
        gate = type(self).__new__(self.__class__)
        gate.__dict__.update(self.__dict__)
        return gate
    
    def relabel(self, mapping: dict):
        # Relabels qubit as per i -> mapping[i].
        new_indices = []
        for i in self.indices:
            new_indices.append(mapping.get(i, None))
            if new_indices[-1] is None:
                raise ValueError(f"Index {i} does not map anywhere.")
        if len(set(new_indices)) != len(new_indices):
            raise ValueError("Indices must be distinct.")
        if min(new_indices) < 0:
            raise ValueError("Non-negative index")
        self.indices = new_indices

    def apply(self, state: np.ndarray) -> np.ndarray:
        N = npq.num_qubits(state)
        if self.matrix is None:
            raise ValueError(f"Matrix representation not given for {self}.")
        gate = npq.expand_gate(self.matrix, N, self.indices)
        if state.ndim == 1:
            return gate @ state
        elif state.ndim == 2:
            return gate @ state @ npq.dagger(gate)
        else:
            raise ValueError("State has wrong dimensions.")


class SingleQubitGate(Gate):
    def __init__(self, index: int, matrix):
        super().__init__([index], matrix)


class TwoQubitGate(Gate):
    def __init__(self, index1: int, index2: int, matrix):
        super().__init__([index1, index2], matrix)


class I(SingleQubitGate):
    def __init__(self, index):
        super().__init__(index, npq.IDTY)

class X(SingleQubitGate):
    def __init__(self, index):
        super().__init__(index, npq.X)

class Y(SingleQubitGate):
    def __init__(self, index):
        super().__init__(index, npq.Y)

class Z(SingleQubitGate):
    def __init__(self, index):
        super().__init__(index, npq.Z)

class H(SingleQubitGate):
    def __init__(self, index):
        super().__init__(index, npq.H)

class RZ(SingleQubitGate):
    def __init__(self, index, angle: float):
        matrix = npq.axis_rotation(angle, [0, 0, 1])
        super().__init__(index, matrix)
        self.angle = angle
    
    def __repr__(self):
        return super().__repr__() + f"({round(self.angle, REPR_DIGITS)})"

class P(SingleQubitGate):
    def __init__(self, index):
        matrix = npq.axis_rotation(np.pi/2, [0, 0, 1])
        super().__init__(index, matrix)

class Pdg(SingleQubitGate):
    def __init__(self, index):
        matrix = npq.axis_rotation(-np.pi/2, [0, 0, 1])
        super().__init__(index, matrix)

class T(SingleQubitGate):
    def __init__(self, index):
        matrix = npq.axis_rotation(np.pi/4, [0, 0, 1])
        super().__init__(index, matrix)

class Tdg(SingleQubitGate):
    def __init__(self, index):
        matrix = npq.axis_rotation(-np.pi/4, [0, 0, 1])
        super().__init__(index, matrix)

class CX(TwoQubitGate):
    def __init__(self, control, target):
        super().__init__(control, target, npq.CX)

    @property
    def control(self):
        return self.indices[0]
    
    @property
    def target(self):
        return self.indices[1]

class CZ(TwoQubitGate):
    def __init__(self, index1, index2):
        super().__init__(index1, index2, npq.CZ)

class SWAP(TwoQubitGate):
    def __init__(self, index1, index2):
        super().__init__(index1, index2, npq.SWAP)

class Insert(SingleQubitGate):
    def __init__(self, index: int, state: State):
        super().__init__(index, state.get().reshape((1, 2)))
        self.state = state
    
    def __repr__(self):
        return super().__repr__() + f"({self.state})"
    
    # Insert gate is a particular special case
    def apply(self, state: np.ndarray) -> np.ndarray:
        insert = self.matrix[0, :]
        i = self.indices[0]
        N = npq.num_qubits(state)
        
        state = npq.tensor(state, insert)
        indices = list(range(i)) + list(range(i+1, N+1)) + [i]
        state = npq.permute_tensor_product(state, indices)
        return state

class M(SingleQubitGate):
    def __init__(self, index: int, theta: float, phi: float, *, result: int=None):
        super().__init__(index, None)
        if result is not None and result not in [0, 1]:
            raise ValueError(f"Measurement results must be from 0 or 1 but {result} was given.")
        self.theta = theta
        self.phi = phi
        self.result = result
    
    # Measurements are a particular special case
    def apply(self, state: np.ndarray) -> tuple[np.ndarray, int]:
        i = self.indices[0]
        N = npq.num_qubits(state)

        rotation = npq.axis_rotation(self.phi, [0, 0, 1]) @ npq.axis_rotation(self.theta, [0, 1, 0])
        eig0 = rotation @ npq.ZERO
        eig1 = rotation @ npq.ONE

        ops = [npq.IDTY]*N
        
        ops[i] = eig0
        res0 = npq.tensor(*ops) @ state
        norm0 = npq.norm(res0)
        
        ops[i] = eig1
        res1 = npq.tensor(*ops) @ state
        norm1 = npq.norm(res1)
        
        s = np.random.choice([0, 1], p=[norm0**2, norm1**2]) if self.result is None else self.result
        
        state = [res0, res1][s] / [norm0, norm1][s]
        return state, s

class MZ(M):
    def __init__(self, index, *, result = None):
        super().__init__(index, 0.0, 0.0, result=result)

class MX(M):
    def __init__(self, index, *, result = None):
        super().__init__(index, np.pi/2, 0.0, result=result)