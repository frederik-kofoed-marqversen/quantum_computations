from .gate_abc import *
from .mps import tensor_svd
from .utils import fourier, rotation, whittaker_shannon
from .states import State
import numpy as np
from numpy.random import Generator as RNG
from scipy.interpolate import RegularGridInterpolator
import logging
from .mps import MPS
logger = logging.getLogger(__name__)


class Insert(SingleModeGate):
    """Insert CV mode"""
    
    def __init__(self, index: int, state: State, *, gkp_epsilon: float=None, **kwargs):
        if kwargs.pop("dagger", None):
            logger.info(type(self).__name__ + "gates ignores adjoint/dagger.")
        super().__init__(index, arg=state, **kwargs)
        
        self.gkp_epsilon = gkp_epsilon
    
    def apply(self, mps: MPS, *, rng: RNG=None, **_):
        # The new mode is inserted into mode initially at index and then split from it 
        # afterwards using a truncated SVD
        # Could be optimised such that the new mode is inserted into the mode at index
        # or index+1 depending on which of those have the smallest bond dimensions.
        state = self.arg.eval(mps.domain, self.gkp_epsilon)
        
        if self.index < 0 or self.index > len(mps):
            raise IndexError(f"Cannot insert mode at index {self.index} for MPS of length {len(mps)}")
        if self.index == 0:
            mps.tensors.insert(0, np.reshape(state, (1, -1, 1)))
            return
        if self.index == len(mps):
            mps.tensors.append(np.reshape(state, (1, -1, 1)))
            return
        
        tensor = np.einsum("i,ajb -> aijb", state, mps[self.index])
        
        logger.info("   doing truncated SVD")
        m1, m2 = tensor_svd(tensor, (0, 1), (2, 3), **self.svd_options, rng_seed=rng)
        
        mps[self.index] = m2
        mps.tensors.insert(self.index, m1)


class SWAP(TwoModeGate):
    """Swap the index of two modes"""
    
    def apply(self, mps: MPS, *, rng: RNG=None, **_):
        m1, m2 = mps[self.left_index], mps[self.right_index]
        res = np.einsum("ijk, klm -> ijlm", m1, m2)
        m1, m2 = tensor_svd(res, [0, 2], [1, 3], **self.svd_options, rng_seed=rng)
        mps[self.left_index], mps[self.right_index] = m1, m2


class BS(TwoModeGate):
    """Beam-splitter gate."""

    def __init__(self, index1, index2, angle: float=np.pi/4, **kwargs):
        super().__init__(index1, index2, arg=angle, **kwargs)
    
    def __repr__(self):
        angle = round(self.arg / np.pi, REPR_DIGITS)
        return type(self).__name__ + f"({angle} * π)" + f"_{self.index1},{self.index2}"

    def apply(self, mps: MPS, *, rng: RNG=None, **_):
        angle = self.arg * (-1)**(self.index1 > self.index2) * (-1)**self.dagger
        qs = mps.domain
        m1, m2 = mps[self.left_index], mps[self.right_index]

        res = np.tensordot(m1, m2, axes=(2,0))
        x, y = np.meshgrid(qs, qs, indexing="ij")
        c, s = np.cos(angle), np.sin(angle)
        x, y = c*x+s*y, -s*x+c*y
        
        rot_2d = lambda arr: RegularGridInterpolator((qs, qs), arr, method='linear', bounds_error=False, fill_value=0)((x, y))
        for a, b in np.ndindex((res.shape[0], res.shape[3])):
            res[a, :, :, b] = rot_2d(res[a, :, :, b])
        
        logger.info("   doing truncated SVD")
        m1, m2 = tensor_svd(res, [0, 1], [2, 3], **self.svd_options, rng_seed=rng)
        mps[self.left_index], mps[self.right_index] = m1, m2
    

class Mq(Measurement):
    """Homodyne measurement along the q-axis"""

    def apply(self, mps: MPS, rng: RNG, **_):
        # Sample measurement result
        qs = mps.domain
        dq = mps.diff
        
        rho = mps.partial_density_mps(self.index)
        distribution = np.real_if_close(np.einsum("ii -> i", rho) * dq)
        if self.result is None:
            s_index = rng.choice(range(len(qs)), p=distribution/np.sum(distribution))
        else:
            s_index = np.argmin(np.abs(qs - self.result))
        s = qs[s_index]
        p = distribution[s_index] / dq

        if len(mps) == 1: # Early escape on edge case
            return s
        
        # Perform measurement
        mode = mps[self.index][:, s_index, :] / np.sqrt(p)
        # Contract the larger of the two inner dimensions of the redundant node
        if np.argmax(mode.shape) == 0 and self.index != 0:
            mps[self.index-1] = np.tensordot(mps[self.index-1], mode, axes=(2, 0))
        else:
            mps[self.index+1] = np.tensordot(mode, mps[self.index+1], axes=(1, 0))
        # Remove the redundant node
        mps.tensors.pop(self.index)

        return MeasurementResult(s, p)


class Mp(Mq):
    """Homodyne measurement along the p-axis"""
    
    def apply(self, mps: MPS, **kwargs):
        mps[self.index] = fourier(mps.domain, mps[self.index], axis=1, inv=True)
        return super().apply(mps, **kwargs)


class Homodyne(Mq):
    """Homodyne measurement along the q-axis rotated by `angle` radians"""
    
    def __init__(self, index, angle: float, result: float=None, **kwargs):
        super().__init__(index, result, arg=angle, **kwargs)
    
    def __repr__(self):
        angle = round(self.arg / np.pi, REPR_DIGITS)
        return type(self).__name__ + f"({angle} * π)" + f"_{self.index}" + (f" = {round(self.result, REPR_DIGITS)}" if self.result else "")
    
    def apply(self, mps: MPS, **kwargs):
        if np.isclose(np.sin(self.arg), 0):
            logger.info("\tsin(angle) ≈ 0 detected: Using Mq gate instead.")
            result = super().apply(mps, **kwargs)
            # Flip result on -q measurements
            result.result *= np.round(np.cos(self.arg))
            return result
        # Maybe use Mp on cos(angle) ≈ 0?
        else:
            mps[self.index] = rotation(mps.domain, mps[self.index], -self.arg, axis=1)
            return super().apply(mps, **kwargs)


class CZ(TwoModeGate):
    """Controlled p-displacement with gain `s`"""

    def __init__(self, index1, index2, s: float=1.0, **kwargs):
        super().__init__(index1, index2, arg=s, **kwargs)
    
    def apply(self, mps: MPS, *, rng: RNG=None, **_):
        qs = mps.domain
        cz = np.exp((-1)**self.dagger*1j*self.arg*np.outer(qs, qs))
        res = np.einsum("ijk, klm, jl -> ijlm", mps[self.left_index], mps[self.right_index], cz, optimize=True)
        
        logger.info("   doing truncated SVD")
        mps[self.left_index], mps[self.right_index] = tensor_svd(res, [0, 1], [2, 3], **self.svd_options, rng_seed=rng)


class CX(TwoModeGate):
    """Controlled q-displacement with gain `s`"""

    def __init__(self, control, target, s: float=1.0, **kwargs):
        super().__init__(control, target, arg=s, **kwargs)
    
    def __repr__(self):
        return Gate.__repr__(self) + f"_{self.index1},{self.index2}"
    
    def apply(self, mps: MPS, *, rng: RNG=None, **_):
        qs = mps.domain
        
        x, y = np.meshgrid(qs, qs, indexing="ij")
        if self.index1 < self.index2:
            # left_index is the control
            x, y = x, y - x * (-1)**self.dagger
        else:
            # right_index is the control
            x, y = x - y * (-1)**self.dagger, y
        res = np.tensordot(mps[self.left_index], mps[self.right_index], (2, 0))
        
        displ_2d = lambda arr: RegularGridInterpolator((qs, qs), arr, method='linear', bounds_error=False, fill_value=0)((x, y))
        for a, b in np.ndindex((res.shape[0], res.shape[3])):
            res[a, :, :, b] = displ_2d(res[a, :, :, b])
        
        logger.info("   doing truncated SVD")
        mps[self.left_index], mps[self.right_index] = tensor_svd(res, [0, 1], [2, 3], **self.svd_options, rng_seed=rng)


class F(SingleModeGate):
    """Fourier gate"""
    
    def apply(self, mps: MPS, **_):
        mps[self.index] = fourier(mps.domain, mps[self.index], axis=1, inv=self.dagger)


class X(SingleModeGate):
    """q-axis displacement by `s`"""
    
    def __init__(self, index, s: float=1.0, **kwargs):
        super().__init__(index, arg=s, **kwargs)
    
    def apply(self, mps: MPS, **_):
        qs = mps.domain
        new_qs = qs - (-1)**self.dagger * self.arg
        mps[self.index] = whittaker_shannon(qs, mps[self.index], new_qs, axis=1)


class Z(SingleModeGate):
    """p-axis displacement by `s`"""
    
    def __init__(self, index, s: float=1.0, **kwargs):
        super().__init__(index, arg=s, **kwargs)
    
    def apply(self, mps: MPS, **_):
        qs = mps.domain
        mps[self.index] = np.einsum("ijk,j -> ijk", mps[self.index], np.exp((-1)**self.dagger * 1j*self.arg * qs))


class D(SingleModeGate):
    """Quadrature displacement by `s` = [s_q, s_p]"""
    
    def __init__(self, index, s: list[float, float], **kwargs):
        if len(s) != 2:
            raise ValueError("s must have exactly 2 elements.")
        super().__init__(index, arg=s, **kwargs)
    
    def apply(self, mps: MPS, **kwargs):
        X(self.index, (-1)**self.dagger * self.arg[0]).apply(mps, **kwargs)
        Z(self.index, (-1)**self.dagger * self.arg[1]).apply(mps, **kwargs)


class P(SingleModeGate):
    """Quadratic phase gate with gain `s`"""
    
    def __init__(self, index, s: float=1.0, **kwargs):
        super().__init__(index, arg=s, **kwargs)
    
    def apply(self, mps: MPS, **_):
        qs = mps.domain
        mps[self.index] = np.einsum("ijk,j -> ijk", mps[self.index], np.exp((-1)**(self.dagger) * 0.5j * self.arg * qs**2))


class S(SingleModeGate):
    """Squeezing gate"""

    def __init__(self, index, r: float, angle: float, **kwargs):
        raise NotImplementedError()
        super().__init__(index, arg=r, **kwargs)
        self.angle = angle
    
    def apply(self, mps, **kwargs):
        raise NotImplementedError()


class Phase(SingleModeGate):
    """Single mode phase rotation gate"""

    def __init__(self, index, angle: float, **kwargs):
        raise NotImplementedError()
        super().__init__(index, arg=angle, **kwargs)
    
    def apply(self, mps, **kwargs):
        raise NotImplementedError()