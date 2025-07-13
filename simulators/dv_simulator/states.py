import numpy as np
from enum import Enum, auto
from . import numpy_quantum as npq

class State(Enum):
    ZERO     = auto()
    ONE      = auto()
    PLUS     = auto()
    MINUS    = auto()
    T        = auto()
    TDG      = auto()
    H        = auto()

    def __repr__(self):
        return self.name
    
    def get(self) -> np.ndarray:
        match self:
            case State.ZERO:
                return npq.ZERO
            case State.ONE:
                return npq.ONE
            case State.PLUS:
                return npq.PLUS
            case State.MINUS:
                return npq.MINUS
            case State.T:
                return np.array([1.0, np.exp(1.0j * np.pi / 4.0)]) * 2**-0.5
            case State.TDG:
                return np.array([1.0, np.exp(-1.0j * np.pi / 4.0)]) * 2**-0.5
            case State.H:
                return np.array([np.cos(np.pi / 8.0), np.sin(np.pi / 8.0)])