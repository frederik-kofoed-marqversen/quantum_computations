import numpy as np
from numpy.random import Generator as RNG
from ..cv_simulator.mps import MPS, tensor_svd
from ..cv_simulator.gates import Insert
from ..cv_simulator.states import State

import logging
logger = logging.getLogger(__name__)

from enum import Enum

PI = np.pi
SQPI = np.sqrt(np.pi)

class GKPBellState(Enum):
    PLUS = 1
    T    = 2
    Tdg  = 3

    def __repr__(self):
        return "GKP_BELL_" + self.name
    
    def __str__(self):
        return self.__repr__()

    def eval(self, qs: np.ndarray, gkp_epsilon: float=None) -> MPS:
        if not isinstance(qs, np.ndarray) or qs.ndim != 1:
            raise TypeError("qs must be a 1D numpy array.")
        # Currently qs must be a sorted list of equidistant points.
        if not np.allclose(np.diff(qs, 2), 0, atol=np.finfo(qs.dtype).eps**0.5):
            raise ValueError("qs is not an arithmetic progression.")
        if gkp_epsilon is not None and gkp_epsilon <= 0:
            raise ValueError("epsilon must be a positive real number")
        
        coeffs = [1, 1]
        match self:
            case GKPBellState.PLUS:
                pass
            case GKPBellState.T:
                coeffs[1] = np.exp(1j*PI/8)
            case GKPBellState.Tdg:
                coeffs[1] = np.exp(-1j*PI/8)

        # Efficient preparation of the qunaught Bell state. 
        # The result is equal to: BS |ø>|ø>.
        bell_tensor = np.zeros((1, len(qs), 2), dtype=complex)
        bell_tensor[:, :, 0] = 2**(-1/4) * coeffs[0] * State.GKP_ZERO.eval(qs, gkp_epsilon)
        bell_tensor[:, :, 1] = 2**(-1/4) * coeffs[1] * State.GKP_ONE.eval(qs, gkp_epsilon)
        
        bell_tensors = [bell_tensor, np.permute_dims(bell_tensor, (2, 1, 0))]
        return MPS(qs, bell_tensors)

class InsertBell(Insert):
    """Insert two-mode CV Bell state"""
    
    def __init__(self, index, state: GKPBellState=GKPBellState.PLUS, *, gkp_epsilon: float=None, **kwargs):
        if not isinstance(state, GKPBellState):
            raise TypeError(f"Expected GKPBellState obj but found {type(state)}")
        super().__init__(index, state, gkp_epsilon=gkp_epsilon, **kwargs)
    
    def apply(self, mps: MPS, *, rng: RNG=None, **_):
        idx = self.index
        bell: MPS = self.arg.eval(mps.domain, self.gkp_epsilon)
        
        # Edge cases
        if idx < 0 or idx > len(mps):
            raise IndexError(f"Cannot insert mode at index {idx} for MPS of length {len(mps)}")
        if idx == 0:
            mps.tensors = bell.tensors + mps.tensors
            return
        if idx == len(mps):
            mps.tensors = mps.tensors + bell.tensors
            return

        # We now have the following networks (in einsum notation)
        # "aib,bjc -> aijc", t1, t2
        # "kd,dl -> kl", b1, b2
        t1, t2 = mps[idx-1], mps[idx]
        b1, b2 = bell[0][0, :, :], bell[1][:, :, 0]  # Remove the redundant dimensions

        # Insert b1 into t1 and collect their right-edges into a single edge
        tb = np.einsum("aib,kd -> aikbd", t1, b1)
        tb = np.reshape(tb, tb.shape[:-2] + (-1,))
        # Restore MPS form
        logger.info("   doing first of two truncated SVDs")
        t1, b1 = tensor_svd(tb, (0, 1), (2, 3), **self.svd_options, rng_seed=rng)
        
        # Same as before, but to the left instead
        tb = np.einsum("dl,bjc -> bdljc", b2, t2)
        tb = np.reshape(tb, (-1,) + tb.shape[2:])
        logger.info("   doing second of two truncated SVDs")
        b2, t2 = tensor_svd(tb, (0, 1), (2, 3), **self.svd_options, rng_seed=rng)

        mps[idx-1] = t1
        mps.tensors.insert(idx, b1)
        mps.tensors.insert(idx + 1, b2)
        mps[idx + 2] = t2