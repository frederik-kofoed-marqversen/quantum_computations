import numpy as np
from enum import Enum, auto

from ..cv_simulator.gates import *

from .utils import PI, SQPI
from .insert_bell import InsertBell, GKPBellState

import logging
logger = logging.getLogger(__name__)

type Syndrome = tuple[int, int]

class MBType(Enum):
    I = auto()
    F = auto()
    P = auto()

    def angles(self):
        match self:
            case MBType.I:
                return [0.0, PI/2]
            case MBType.F:
                return [PI/4, -PI/4]
            case MBType.P:
                return [0.0, np.arctan(2)]

class MB2Type(Enum):
    II   = auto()
    FF   = auto()
    PP   = auto()
    PPdg = auto()
    CZ   = auto()
    SWAP = auto()

    def angles(self):
        match self:
            case MB2Type.II:
                return [0.0, 0.0, PI/2, PI/2]
            case MB2Type.FF:
                return [PI/4, PI/4, -PI/4, -PI/4]
            case MB2Type.PP:
                return [0.0, 0.0, np.arctan(2), np.arctan(2)]
            case MB2Type.PPdg:
                return [0.0, 0.0, np.arctan(2), -np.arctan(2)]
            case MB2Type.CZ:
                return [0.0, 0.0, np.arctan(2), -np.arctan(2)]
            case MB2Type.SWAP:
                return [-PI/2, 0.0, 0.0, -PI/2]

class MeasurementBased(ABC):
    """Abstract base class MB GKP gates."""

    def __init__(self, indices: list[int], type: MBType | MB2Type, epsilon: float=None, *, dagger: bool=False, **kwargs):
        """
        Initialize the gate. Unexpected keyword arguments are logged.
        
        :param type: The argument for parametrised gates.
        :param dagger: Whether to initialize the adjoint gate.
        :param **kwargs: Include options for svd trunctation.
        """
        self.indices = indices
        self.epsilon = epsilon
        self.type = type
        self.dagger = dagger
        self.svd_options = {key: kwargs.pop(key) for key in SVD_OPTIONS if key in kwargs}

        if kwargs:
            logger.warning(f"{type(self).__name__} recieved unexpected keyword arguments: {kwargs.keys()}")
    
    def angles(self) -> list[float]:
        return np.array(self.type.angles()) * (-1)**self.dagger
    
    @abstractmethod
    def compile(self) -> list[Gate]:
        """
        Compile gate into a sequence of executable gates.
        """
    
    @abstractmethod
    def compute_syndrome(self, results: list[float]) -> tuple[list[Syndrome], list[int]]:
        """
        Computes the syndrome given measurement results, together with a list mapping each syndrome
        to a mode index in the final output state. The `results` should be ordered consistently wrt.
        the order of measurements supplied by `self.compile()`.
        """


class MBSingleMode(MeasurementBased):
    """
    Error corrected, single mode, and parametrised Gaussian gate as defined in: 
    B. W. Walshe et al. “Continuous-variable gate teleportation and bosonic-
    code error correction”. In: Physical Review A 102.6 (Dec. 15, 2020), p. 062411.
    doi: 10.1103/PhysRevA.102.062411.
    """
    
    def __init__(self, index: int, type: MBType, epsilon: float=None, *, results: tuple[float, float]=None, **kwargs):
        super().__init__([index], type, epsilon, **kwargs)
        self.results = results if results is not None else (None, None)
        if len(self.results) != 2:
            raise ValueError("Results list must have exactly 2 elements.")

    def compile(self):
        idx = self.indices[0]
        angles = self.angles()
        return [
            InsertBell(idx+1, gkp_epsilon=self.epsilon, **self.svd_options),
            BS(idx, idx+1, **self.svd_options),
            Homodyne(idx, angles[0], result=self.results[0]),
            Homodyne(idx, angles[1], result=self.results[1]),
        ]
    
    """Computes the syndrome (n, m) which is to be fixed by applying X(n√π) Z(m√π)"""
    def compute_syndrome(self, results: list[float]) -> list[Syndrome]:
        if len(results) != 2:
            raise ValueError("Exactly two measurement results are needed.")
        # Fetch measurements and compute byproduct displacement mu
        ta, tb = self.angles()
        ma, mb = results
        # Displacement - multiplied by -1j since we measure angles from q-axis instead of p-axis as they do in article
        mu = 1j *(ma * np.exp(1j*tb) + mb * np.exp(1j*ta)) / np.sin(ta - tb)
        # Equivalant quadrature displacement vector
        mu = np.array([mu.real, mu.imag]) * 2**0.5
        # Compute the logical syndrome
        syndrome = np.round(mu / SQPI) % 2
        syndrome = tuple(map(int, syndrome))
        return [syndrome], self.indices


class MBTwoMode(MeasurementBased):
    """
    Error corrected, two mode, and parametrised Gaussian gate as defined in: 
    B. W. Walshe et al. Streamlined quantum computing with macronode cluster 
    states. Jan. 4, 2022. doi: 10.48550/arXiv.2109.04668. arXiv: 2109.04668.
    
    Ordering of `angles` and `results` is: [a, c, b, d], with labels a,b,c,d assigned as in the paper.
    In particular a is the measurement on the left-most/smallest input index, and b is the measured 
    ancilla next to it.
    """
    
    def __init__(self, index1: int, index2: int, type: MB2Type, epsilon: float=None, *, results: tuple[float, float, float, float]=None, **kwargs):
        if abs(index1 - index2) != 1:
            raise ValueError(f"{type(self).__name__} can only be applied to neighbours, but indices: {(index1, index2)} were given.")
        results = results if results is not None else (None, None, None, None)
        if len(results) != 4:
            raise ValueError("Results list must have exactly 4 elements.")
        
        super().__init__(sorted([index1, index2]), type, epsilon, **kwargs)
        self.results = results
    
    def compile(self):
        idx = min(self.indices)
        ta, tc, tb, td = self.angles()
        ma, mc, mb, md = self.results
        return [
            InsertBell(idx, gkp_epsilon=self.epsilon, **self.svd_options),
            InsertBell(idx+4, gkp_epsilon=self.epsilon, **self.svd_options),

            BS(idx+2, idx+1, **self.svd_options),
            BS(idx+3, idx+4, **self.svd_options),
            
            BS(idx+2, idx+3, **self.svd_options),
            
            Homodyne(idx+2, ta, result=ma),
            Homodyne(idx+2, tc, result=mc),
            
            BS(idx+1, idx+2, **self.svd_options),

            Homodyne(idx+1, tb, result=mb),
            Homodyne(idx+1, td, result=md),
        ]
    
    """
    Computes the syndrome (n1, m1), (n2, m2) which is to be fixed by applying X(n1√π) Z(m1√π) ⊗ X(n2√π) Z(m2√π)
    """
    def compute_syndrome(self, results: list[float]) -> list[Syndrome]:
        if len(results) != 4:
            raise ValueError("Exactly two measurement results are needed.")
        
        # Fetch measurements and compute byproduct displacements mu1, mu2
        ta, tc, tb, td = self.angles()
        ma, mc, mb, md = results

        # Compute complex displacements
        mu_ab = 1j * (ma * np.exp(1j*tb) + mb * np.exp(1j*ta)) / np.sin(ta - tb)
        mu_cd = 1j * (mc * np.exp(1j*td) + md * np.exp(1j*tc)) / np.sin(tc - td)
        
        mu1 = (mu_cd + mu_ab) # / 2**0.5  # Constant factor is cancelled in next step
        mu2 = (mu_cd - mu_ab) # / 2**0.5
        # Equivalant quadrature displacement vectors
        mu1 = np.array([mu1.real, mu1.imag]) # * 2**0.5
        mu2 = np.array([mu2.real, mu2.imag]) # * 2**0.5
        # Compute the logical syndrome
        syndrome1 = np.round(mu1 / SQPI) % 2
        syndrome2 = np.round(mu2 / SQPI) % 2
        # Convert from numpy to python
        syndrome1 = tuple(map(int, syndrome1))
        syndrome2 = tuple(map(int, syndrome2))

        return [syndrome1, syndrome2], self.indices


class MBI(MBSingleMode):
    """Error correction using the Knill method"""
    
    def __init__(self, index, epsilon: float=None, *, results: tuple[float, float]=None, **kwargs):
        super().__init__(index, MBType.I, epsilon=epsilon, results=results, **kwargs)


# Type alias
type GKPEC = MBI


class MBF(MBSingleMode):
    """Error corrected Fourier gate"""
    
    def __init__(self, index, epsilon: float=None, *, results: tuple[float, float]=None, **kwargs):
        super().__init__(index, MBType.F, epsilon=epsilon, results=results, **kwargs)


class MBP(MBSingleMode):
    """Error corrected P gate"""
    
    def __init__(self, index, epsilon: float=None, *, results: tuple[float, float]=None, **kwargs):
        super().__init__(index, MBType.P, epsilon=epsilon, results=results, **kwargs)


class MBSWAP(MBTwoMode):
    """Error corrected controlled Z gate"""
    
    def __init__(self, index1: int, index2: int, epsilon: float=None, *, results: tuple[float, float, float, float]=None, **kwargs):
        super().__init__(index1, index2, MB2Type.SWAP, epsilon=epsilon, results=results, **kwargs)


class MBCZ(MBTwoMode):
    """Error corrected controlled Z gate"""
    
    def __init__(self, index1: int, index2: int, epsilon: float=None, *, results: tuple[float, float, float, float]=None, **kwargs):
        super().__init__(index1, index2, MB2Type.CZ, epsilon=epsilon, results=results, **kwargs)


class MBT(MBSingleMode):
    """Measurement-based implementation of the non-Clifford (non-Gaussian) T gate."""
    
    def __init__(self, index, epsilon: float=None, *, results: tuple[float, float]=None, **kwargs):
        super().__init__(index, MBType.I, epsilon=epsilon, results=results, **kwargs)
    
    def compile(self):
        idx = self.indices[0]
        bell = GKPBellState.T if not self.dagger else GKPBellState.Tdg
        angles = MBType.I.angles()
        
        return [            
            InsertBell(idx+1, bell, gkp_epsilon=self.epsilon, **self.svd_options),
            BS(idx, idx+1, **self.svd_options),
            Homodyne(idx, angles[0], result=self.results[0]),
            Homodyne(idx, angles[1], result=self.results[1]),
        ]