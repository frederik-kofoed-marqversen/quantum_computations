import numpy as np
from .gates import Gate
from .states import State
from .numpy_quantum import tensor

class ClassicalControl():
    def __init__(self, gate: Gate, positive_indices: list[int]=[], negative_indices: list[int]=[]):
        self.gate = gate
        self.indices = gate.indices
        self._pos = positive_indices
        self._neg = negative_indices
    
    def __repr__(self):
        return f"Classical control: {self.gate}"
    
    def eval(self, observables: list[bool]) -> bool:
        return all(observables[i] for i in self._pos) and all(not observables[i] for i in self._neg)

def parse_state(state: np.ndarray|list[State]|None) -> np.ndarray:
    match state:
        case None:
            return np.ones((1,))
        case np.ndarray():
            return state
        case list() if all(isinstance(item, State) for item in state):  
            return tensor(*(s.get() for s in state))
        case _:
            raise TypeError("Unsupported input type")

class Simulator:
    def __init__(self, circuit: list[Gate], rng_seed: int=None):
        self.circuit: list[Gate] = circuit
        self.results: list[int] = None
        self._rng = np.random.default_rng(rng_seed)

    def run(self, initial_state: np.ndarray|list[State]=None) -> np.ndarray:
        self.results = []
        state = parse_state(initial_state)

        for gate in self.circuit:
            if isinstance(gate, ClassicalControl):
                if gate.eval(self.results):
                    gate = gate.gate
                else:
                    continue
            
            output = gate.apply(state)
            if isinstance(output, tuple):
                state = output[0]
                self.results.append(output[1])
            else:
                state = output
        return state