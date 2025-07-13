import numpy as np
from mpmath import jtheta
from scipy.special import hermite, factorial
from enum import Enum, auto

PI = np.pi
SQPI = np.sqrt(np.pi)

class State(Enum):
    GKP_ZERO  = auto()
    GKP_ONE   = auto()
    GKP_PLUS  = auto()
    GKP_MINUS = auto()
    GKP_T     = auto()
    GKP_TDG   = auto()
    GKP_H     = auto()
    VACUUM    = auto()
    QUNAUGHT  = auto()

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.__repr__()

    def eval(self, qs: np.ndarray, gkp_epsilon: float=None) -> np.ndarray:
        if not isinstance(qs, np.ndarray) or qs.ndim != 1:
            raise TypeError("qs must be a 1D numpy array.")
        # Currently qs must be a sorted list of equidistant points.
        if not np.allclose(np.diff(qs, 2), 0, atol=np.finfo(qs.dtype).eps**0.5):
            raise ValueError("qs is not an arithmetic progression.")
        if gkp_epsilon is not None and gkp_epsilon <= 0:
            raise ValueError("epsilon must be a positive real number")
        
        gkp_coeffs = None
        result = None
        match self:
            case State.GKP_ZERO:
                gkp_coeffs = [1, 0]
            case State.GKP_ONE:
                gkp_coeffs = [0, 1]
            case State.GKP_PLUS:
                gkp_coeffs = [1, 1]
            case State.GKP_MINUS:
                gkp_coeffs = [1, -1]
            case State.GKP_T:
                gkp_coeffs = [1, np.exp(1j*PI/4)]
            case State.GKP_TDG:
                gkp_coeffs = [1, np.exp(-1j*PI/4)]
            case State.GKP_H:
                gkp_coeffs = [np.cos(PI/8), np.sin(PI/8)]
            case State.VACUUM:
                return vacuum(qs)
            case State.QUNAUGHT:
                if gkp_epsilon is None:
                    raise ValueError("Evaluating qunaught states require a gkp_epsilon.")
                result = comb_sym(qs, gkp_epsilon, np.sqrt(2*PI))
        
        if gkp_coeffs is not None:
            if gkp_epsilon is None:
                    raise ValueError("Evaluating gkp states require a gkp_epsilon.")
            result = gkp_sym(qs, gkp_epsilon, gkp_coeffs)
        
        # Normalise
        dq = abs(qs[-1] - qs[0]) / (len(qs) - 1)
        result /= np.sqrt(np.real_if_close(np.sum(result * np.conjugate(result)) * dq))
        return result


def eval_gkp_state(qs: np.ndarray, epsilon: float, coefficients: tuple[complex, complex]) -> np.ndarray:
    # Evaluate
    result = gkp_sym(qs, epsilon, coefficients)
    # Normalise
    dq = abs(qs[-1] - qs[0]) / (len(qs) - 1)
    result /= np.sqrt(np.real_if_close(np.sum(result * np.conjugate(result)) * dq))
    return result


# STATE DEFINITIONS
# These all allow for their input `q` to be both a float or a numpy array and will return a similar sized output

# note that the rotated eigenstate is symmetric in `q` and `x` and so can be used either way
rotated_eigenstate = lambda q, x, theta: (2*PI*abs(np.sin(theta)))**(-0.5) * np.exp(-1j * (np.cos(theta)*(q*q + x*x)/2 - x*q) / np.sin(theta))
momentum_eigenstate = lambda q, p: np.exp(-1j*q*p) / SQPI # if np.isscalar(p) else np.exp(-1j*np.outer(q, p)) / SQPI

_delta_theta = lambda delta, theta: np.sqrt((np.cos(theta) * delta)**2 + (np.sin(theta) / delta)**2)
squeezed_coherent = lambda q, alpha, r, theta: (PI * _delta_theta(np.exp(r), theta)**2)**(-1/4) * np.exp(-0.5*((q - alpha.real)/_delta_theta(np.exp(r), theta))**2 * (1 - 1j*np.sinh(2*r)*np.sin(2*theta)) + 1j*alpha.imag*q)
# aliases for well-known states (slower than optimal execution due to lazy implementation)
vacuum = lambda q: squeezed_coherent(q, 0, 0, 0)
coherent = lambda q, alpha: squeezed_coherent(q, alpha, 0, 0)
squeezed_vac = lambda q, r: squeezed_coherent(q, 0, r, 0)

fock_state = lambda q, n: hermite(n)(q) * np.exp(-q**2/2) * (2**n * factorial(n) * SQPI)**(-0.5)

# GKP STATES
# Helper functions
def theta(z, tau):  # Jacobi theta function (of the third kind)
    # The definition in mpmath differ with that of wiki by z -> PI*z
    z = PI * z
    q = np.exp(1j*PI*tau)
    return float(jtheta(3, z, q))
theta = np.vectorize(theta)  # Make `theta` work for ndarrays. This is unfortunately a slow implementation

def modified_theta(a, b, z, tau):
    return np.exp(PI*1j*tau*a**2 + 2j*PI*a*(z+b)) * theta(z + a*tau + b, tau)

def gaussians(s, delta_sq, alpha=2*SQPI):
    # Equally spaced normalised Gaussian functions with width delta_sq
    # The peaks are located at alpha * n, default is alpha=2*\sqrt{\pi} corresponding to square GKP states
    return theta(s/alpha, 2j*PI*delta_sq/alpha**2) / alpha

# non-normalised GKP states as defined in "Equivalence of approximate Gottesman-Kitaev-Preskill codes" by Matsuura et al.
# the argument `state` are the logical coefficients (maybe change to Bloch sphere definition?)
# maybe also include different types of approximations
gkp = lambda q, kappa, delta, state=[1, 0]: np.exp(-q**2/2 / ((1+delta**2*kappa**2)/kappa**2)) * sum(c * modified_theta(0, mu/2, -q/(2*SQPI*(1 + kappa**2*delta**2)), 0.5j*delta**2/(1 + kappa**2*delta**2)) for mu, c in enumerate(state))
gkp_sym = lambda q, epsilon, state=[1, 0]: np.exp(-np.tanh(epsilon)*q**2/2) * sum(c * modified_theta(0, mu/2, -q/(2*SQPI*np.cosh(epsilon)), 1j*np.tanh(epsilon)/2) for mu, c in enumerate(state))
# Gaussian envelope on Gaussian comb with distance given by alpha
comb = lambda q, kappa, delta, alpha: np.exp(-q**2/2 / ((1+delta**2*kappa**2)/kappa**2)) * modified_theta(0, 0, -q/(alpha*(1 + kappa**2*delta**2)), 1j*delta**2/(1 + kappa**2*delta**2))
comb_sym = lambda q, epsilon, alpha: np.exp(-np.tanh(epsilon)*q**2/2) * modified_theta(0, 0, -q/(alpha*np.cosh(epsilon)), 1j*np.tanh(epsilon))
# The special qunaught state
qunaught = lambda q, epsilon: comb_sym(q, epsilon, np.sqrt(2*PI))