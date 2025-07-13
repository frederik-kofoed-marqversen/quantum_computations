from ..dv_simulator.gates import Gate as DVGate
from ..dv_simulator import gates as dv_gates
from ..dv_simulator.simulator import ClassicalControl
from ..dv_simulator.states import State as DVState
from ..cv_simulator.gate_abc import Gate as CVGate
from ..cv_simulator.states import State as CVState
from .gates import *
from bisect import insort

IMPLEMENTABLES = (dv_gates.I, dv_gates.H, dv_gates.P, dv_gates.Pdg, dv_gates.T, dv_gates.Tdg, dv_gates.CZ, dv_gates.SWAP)
PAULIS = (dv_gates.I, dv_gates.X, dv_gates.Y, dv_gates.Z)

def parse_to_mps(state: MPS|list[DVState]|None, epsilon: float, qs: np.ndarray) -> MPS:
    match state:
        case None:
            return MPS(qs, [])
        case MPS():
            return state
        case list() if all(isinstance(item, DVState) for item in state):  
            return MPS(qs, [state_transpile(s).eval(qs, epsilon) for s in state])
        case _:
            raise TypeError("Unsupported input type")

def state_transpile(state: DVState) -> CVState:
    match state:
        case DVState.ZERO:
            return CVState.GKP_ZERO
        case DVState.ONE:
            return CVState.GKP_ONE
        case DVState.PLUS:
            return CVState.GKP_PLUS
        case DVState.MINUS:
            return CVState.GKP_MINUS
        case DVState.T:
            return CVState.GKP_T
        case DVState.TDG:
            return CVState.GKP_TDG
        case DVState.H:
            return CVState.GKP_H

def gate_transpile(gate: DVGate, **kwargs) -> MeasurementBased:
    dagger = (type(gate) in (dv_gates.Pdg, dv_gates.Tdg)) ^ kwargs.pop("dagger", False)
    gate_type = None
    match type(gate):
        case dv_gates.I:
            gate_type = MBI
        case dv_gates.H:
            gate_type = MBF
        case dv_gates.P:
            gate_type = MBP
        case dv_gates.Pdg:
            gate_type = MBP
        case dv_gates.T:
            gate_type = MBT
        case dv_gates.Tdg:
            gate_type = MBT
        case dv_gates.CZ:
            gate_type = MBCZ
        case dv_gates.SWAP:
            gate_type = MBSWAP
        case _:
            raise ValueError(f"Gate {gate} not implementable in MB GKP circuits.")
    return gate_type(*gate.indices, dagger=dagger, **kwargs)

class Layer:
    def __init__(self, N: int):
        self._N = N
        self._occupied: list[bool] = [False] * N
        self.gates: list[DVGate | ClassicalControl] = []
        self.paulis: list[tuple[int, int]] = [[0, 0] for _ in range(N)]
    
    # Return shallow copy
    def copy(self) -> 'Layer':
        result = Layer(self._N)
        result.gates = self.gates.copy()
        result.paulis = self.paulis.copy()
        return result
    
    # Add identity gates to all available qubits
    def fill(self):
        for i in range(self._N):
            if not self.get_gate(i):
                self._insert_gate(dv_gates.I(i))

    # Get gate applied to qubit at `index`.
    def get_gate(self, index: int) -> CVGate | None:
        for gate in self.gates:
            if index in gate.indices:
                return gate
        return None
    
    # Check if gate is applied to qubits that are already occupied in layer
    def occupied(self, indices: list[int]) -> bool:
        return any(self._occupied[i] or self.paulis[i] != [0, 0] for i in indices)
    
    # Try to add gate to layer and return bool indicating success
    def add_gate(self, gate: DVGate | ClassicalControl) -> bool:
        if self.occupied(gate.indices):
            return False
        self._insert_gate(gate)
        return True
    
    # Insert gate without checks
    def _insert_gate(self, gate: DVGate):
        for i in gate.indices:
            self._occupied[i] = True
        insort(self.gates, gate, key=lambda g: min(g.indices))
    
    # Add Pauli operator to layer
    def add_pauli(self, index: int, pauli: tuple[int, int]):
        self.paulis[index][0] = (self.paulis[index][0] + pauli[0]) % 2
        self.paulis[index][1] = (self.paulis[index][1] + pauli[1]) % 2

class MBGKPCircuit:
    def __init__(self, N: int):
        self._N = N
        self._layers: list[Layer] = [Layer(N)]
    
    def to_string(self) -> str:
        result = ""
        for row_num in range(self._N):
            row = ""
            for layer in self._layers:
                gate = layer.get_gate(row_num)
                if isinstance(gate, ClassicalControl):
                    row += (f"'{gate.gate}'").ljust(8)
                else:
                    row += str(gate).ljust(8)
                row += " "
                row += str(layer.paulis[row_num])
                row += " | "
            result += row[:-3] + "\n"
        return result[:-1]
    
    @staticmethod
    def transpile(gates: list[DVGate], N: int=None) -> 'MBGKPCircuit':
        if N is None:
            N = max(max(gate.indices) for gate in gates) + 1
        circ = MBGKPCircuit(N)
        for gate in gates:
            circ.add_gate(gate)
        return circ
    
    def depth(self) -> int:
        return len(self._layers)

    def count(self) -> int:
        return sum(len(layer.gates) for layer in self._layers)
    
    def fill(self):
        for layer in self._layers:
            layer.fill()
    
    def add_gate(self, gate: DVGate):
        if any(i < 0 or i >= self._N for i in gate.indices):
            raise ValueError(f"Cannot add {gate} to MBGKPCircuit with {self._N} qubits.")
        if len(gate.indices) > 2 :
            raise ValueError(f"Only single- and two-mode gates available, but gate {gate} was given.")
        if len(gate.indices) == 2 and abs(gate.indices[0] - gate.indices[1]) != 1:
            raise ValueError(f"Only nearest neighbour interactions available, but gate {gate} was given.")

        match type(gate):
            case t if t in IMPLEMENTABLES:
                self._add_gate(gate)
                if isinstance(gate, dv_gates.T):
                    self._add_gate(ClassicalControl(dv_gates.P(gate.indices[0]), [-self._N]))
                elif isinstance(gate, dv_gates.Tdg):
                    self._add_gate(ClassicalControl(dv_gates.Pdg(gate.indices[0]), [-self._N]))
            case t if t in PAULIS:
                self._add_pauli(gate)
            case _:
                raise ValueError(f"Gate {gate} not implementable in MB GKP circuits.")
    
    # Find index of first occupied layer from the back
    def _first_occupied(self, indices: list[int]):
        for i in range(len(self._layers)):
            index = -(i+1)
            if self._layers[index].occupied(indices):
                return index
        # Only happens if no layer is occupied
        return None

    # Add gate to first possible layer and add new layer if needed.
    def _add_gate(self, gate: DVGate | ClassicalControl):
        index = self._first_occupied(gate.indices)
        if index is None:
            # All layers available
            index = -1
        elif index == -1:
            # Rightmost layer is already occupied => add layer
            self._layers.append(Layer(self._N))
            index = -2
        self._layers[index + 1].add_gate(gate)

    def _add_pauli(self, gate: DVGate):
        pauli = None
        match type(gate):
            case dv_gates.X:
                pauli = [1, 0]
            case dv_gates.Y:
                pauli = [1, 1]
            case dv_gates.Z:
                pauli = [0, 1]
            case _:
                raise ValueError("Should never get here...")
        index = self._first_occupied(gate.indices)
        if index is None:
            index = 0
        self._layers[index].add_pauli(gate.indices[0], pauli)