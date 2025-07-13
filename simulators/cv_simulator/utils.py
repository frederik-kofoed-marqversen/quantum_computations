import numpy as np
import scipy.fft as fft
import logging
logger = logging.getLogger(__name__)

def wigner(state: np.ndarray, q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("Evaluation of Wigner function not yet implemented")

def whittaker_shannon(xs: np.ndarray, ys: np.ndarray, new_xs: np.ndarray, axis: int = 0) -> np.ndarray:
    # Whittaker–Shannon interpolation / finite bandwidth interpolation / sinc interpolation
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)
    sinc = np.sinc((new_xs[:, np.newaxis] - xs[np.newaxis, :]) / dx)
    
    res = ys
    res = np.tensordot(sinc, res, [1, axis])
    res = np.moveaxis(res, 0, axis)

    return res

interpolate = whittaker_shannon

def rotation(qs: np.ndarray, tensor: np.ndarray, theta: float, axis: int=0, new_qs: np.ndarray=None) -> np.ndarray:
    """Applies phase rotation by angle:`theta` to index:`axis` defined on positions:`qs` and evaluates the result on the positions `new_qs`.
    If `new_qs` are None, the result is evaluated at the same positions as given by `qs`.
    The resulting tensor will have the same indices as `tensor` but whith the dimension of index:axis equal to len(new_qs)"""
    if new_qs is None:
        new_qs = qs
    
    q_span = np.array([min(qs), max(qs)])
    dq = (q_span[1] - q_span[0]) / (len(qs) - 1)

    # rotated_eigenstate = (2*PI*abs(np.sin(theta)))**(-0.5) * np.exp(-1j * (np.cos(theta)*(q*q + x*x)/2 - x*q) / np.sin(theta))
    exponent = np.cos(theta)*((qs**2)[:, np.newaxis] + (new_qs**2)[np.newaxis, :])/2 - np.outer(qs, new_qs)
    rotated_eigenstates = (2*np.pi*abs(np.sin(theta)))**(-0.5) * np.exp(exponent / (1j*np.sin(theta)))
    
    res = tensor
    res = np.tensordot(rotated_eigenstates, res, [0, axis])
    res = np.moveaxis(res, [0], [axis]) * dq
    return res

def fourier(qs: np.ndarray, tensor: np.ndarray, axis: int=0, ps: np.ndarray=None, inv: bool=False) -> np.ndarray:
    """Optimised version of rotation(theta=pi/2) using FFT. The result is evaluated 
    at points `ps`. If `None` then `qs` are used. 
    Evalutation/interpolation is performed by sinc interpolation. We do this instead of zero
    padding. Although this might be slightly inefficient in some cases, it saves memory."""
    if ps is None:
        ps = qs
    
    # The action of the Fourier operator is the inverse Transform on the wave function
    # F \ket{\psi} = \ket{F^{-1}[\psi]}
    _ps, res = iCFT(qs, tensor, axis=axis) if not inv else CFT(qs, tensor, axis=axis)

    if ps[-1] - ps[0] > _ps[-1] - _ps[0]:
        logger.warning("Evaluation outside of the Nyquist bandwidth might have unintended sideeffects.")

    # The Fourier transform is periodic due to sampling
    ps = (ps - _ps[-1]) % (_ps[-1] - _ps[0]) + _ps[0]
    res = whittaker_shannon(_ps, res, ps, axis=axis)
    return res

def CFT(qs: np.ndarray, tensor: np.ndarray, axis: int=0) -> tuple[np.ndarray, np.ndarray]:
    """Numerical computation of the quantum continuous Fourier transform
    defined as: F(omega) = (2*pi)^{-1/2} int dq f(q) e^{-ipq}"""
    
    N = tensor.shape[axis]
    T = (qs[-1] - qs[0]) * N / (N-1)
    ps = fft.fftshift(fft.fftfreq(N, d=T/(N * 2 * np.pi)))
    fs_hat = fft.fftshift(fft.fft(tensor, axis=axis), axes=axis)
    
    phase = T/(N * np.sqrt(2*np.pi)) * np.exp(-1j*ps*qs[0])
    dims = [1]*fs_hat.ndim
    dims[axis] = -1
    fs_hat = fs_hat * np.reshape(phase, dims)
    
    return ps, fs_hat

def iCFT(qs: np.ndarray, tensor: np.ndarray, axis: int=0) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of CFT"""
    ps, fs_hat = CFT(qs, tensor, axis=axis)

    ps = np.flip(-ps)
    fs_hat = np.flip(fs_hat, axis=axis)
    
    return ps, fs_hat