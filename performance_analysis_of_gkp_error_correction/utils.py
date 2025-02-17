import numpy as np
from matplotlib import pyplot as plt
from mpmath import jtheta
from scipy.signal import fftconvolve
from scipy.fft import fftshift, fftfreq, fft
from itertools import product
from functools import reduce

def theta_fun(z, tau):  # Jacobi theta function (of the third kind)
    # The definition in mpmath differ with that of wiki by z -> pi*z
    z = np.pi * z
    q = np.exp(1j*np.pi*tau)
    return float(jtheta(3, z, q))
theta_fun = np.vectorize(theta_fun)  # Make `theta` work for ndarrays. This is a lazy and suboptimal implementation

def modified_theta(a, b, z, tau):
    return np.exp(np.pi*1j*tau*a**2 + 2j*np.pi*a*(z+b)) * theta_fun(z + a*tau + b, tau)

# For physical GKP states, we use (approximation 3) from T. Matsuura et al. “Equivalence of approximate Gottesman-Kitaev-Preskill codes”. 
# In: Physical Review A 102.3 (Sept. 15, 2020), p. 032408. doi: 10.1103/PhysRevA.102.032408. arXiv: 1910.08301[quant-ph].
# We use the notation epsilon = Delta^2
gkp = lambda q, epsilon, state=[1, 0]: np.exp(-np.tanh(epsilon)*q**2/2) * sum(c * modified_theta(0, mu/2, -q/(2*np.sqrt(np.pi)*np.cosh(epsilon)), 1j*np.tanh(epsilon)/2) for mu, c in enumerate(state))
comb = lambda q, epsilon, alpha: np.exp(-np.tanh(epsilon)*q**2/2) * modified_theta(0, 0, -q/(alpha*np.cosh(epsilon)), 1j*np.tanh(epsilon))

def normalise(qs, state):
    norm = np.sqrt(np.trapezoid(state * state.conj(), qs))
    return state / norm

def fourier(qs, state):
    N = len(qs)
    T = (qs[-1] - qs[0]) * N / (N-1)
    ps = fftshift(fftfreq(N, d=T/(N * 2 * np.pi)))
    fs = fftshift(fft(state))
    
    phase = T/(N * np.sqrt(2*np.pi)) * np.exp(-1j*ps*qs[0])
    fs = fs * phase

    new_ps = (qs - ps[-1]) % (ps[-1] - ps[0]) + ps[0]

    # sinc interpolation
    dp = (ps[-1] - ps[0]) / (len(ps) - 1)
    sinc = np.sinc((new_ps[:, np.newaxis] - ps[np.newaxis, :]) / dp)
    result = sinc @ fs
    
    return result

def gkp_project_asym(qs, state, zero, axis: int=0) -> tuple[np.ndarray, np.ndarray]:
    dq = (qs[-1] - qs[0]) / len(qs)
    plus = fourier(qs, zero)
    state = np.moveaxis(state, [axis], [0])
    state = np.einsum("i...,i -> i...", state, plus)
    state = fftconvolve(state, zero[(...,) + (np.newaxis,)*(len(state.shape)-1)], axes=[0], mode="same")
    state *= dq  # Since we use discrete convolution to estimate a continuous one we need to include "dq" to get integration

    state = np.moveaxis(state, [0], [axis])
    return state

def gkp_project_sym(qs, state, zero, one, axis: int=0) -> tuple[np.ndarray, np.ndarray]:
    dq = (qs[-1] - qs[0]) / len(qs)
    bell = (np.outer(zero, zero) + np.outer(one, one)) * 2**-0.5
    
    state = np.tensordot(bell, state, axes=(1, axis)) * dq / np.sqrt(2*np.pi)
    state = np.moveaxis(state, [0], [axis])
    return state

def full_logical_density(qs: np.ndarray, state: np.ndarray):
    # Logical density matrix as defined in appendix D in M. H. Shaw et al. "Logical Gates and Read-Out of Superconducting 
    # Gottesman-Kitaev-Preskill Qubits." Apr. 5, 2024. arXiv: 2403.02396[quant-ph].

    dq = (qs[-1] - qs[0]) / len(qs)
    q_thingy = qs[:, np.newaxis] - qs[np.newaxis, :]

    # construct Pauli measurement operators
    Im = np.identity(len(qs))
    Xm = np.zeros((len(qs), len(qs)))
    Zm = np.zeros((len(qs), len(qs)))
    for n, m in enumerate(range(1, 20, 2)):
        # 2n+1 = m
        coeff = (-1)**(n%2) * 2 / (m * np.pi)
        
        T1 = np.sinc((q_thingy - m*np.sqrt(np.pi)) / dq)
        T2 = np.sinc((q_thingy + m*np.sqrt(np.pi)) / dq)
        Xm += coeff * (T1 + T2)

        # T1 = np.diag(np.exp(1j * SQPI * (+m) * qs))
        # T2 = np.diag(np.exp(1j * SQPI * (-m) * qs))
        T1plusT2 = np.diag(2 * np.cos(np.sqrt(np.pi) * m * qs))
        Zm += coeff * T1plusT2
    Ym = 1j * Xm @ Zm

    # Pauli measurement operators
    Pms = [Im, Xm, Ym, Zm]
    # Pauli logical operators
    Ps = [
        np.array([[1, 0], [0, 1]]),    # I
        np.array([[0, 1], [1, 0]]),    # X
        np.array([[0, -1j], [1j, 0]]), # Y
        np.array([[1, 0], [0, -1]]),   # Z
    ]
    
    # Construct vector representation
    N = state.ndim
    result = np.zeros((2**N, 2**N), dtype=complex)
    for index in product(*[[0, 1, 2, 3],]*N):
        ket = state
        for i in range(N):
            ket = np.tensordot(ket, Pms[index[i]], axes=(0, 1))
        bra = state.conj()
        coeff = (dq/2)**N * np.tensordot(bra, ket, axes=N)
        
        logical_pauli = reduce(np.kron, [Ps[i] for i in index], 1)
        result += coeff * logical_pauli
    return result

def logical_fidelity(qs, state):
    rho = full_logical_density(qs, state)
    rho /= np.trace(rho)
    return np.trace(rho@rho).real


# Helper functions for plotting

def get_tickmarks(min, max, alt_labels=False) -> tuple[list, list]:
    ns = np.arange(round(min/np.sqrt(np.pi)), round(max/np.sqrt(np.pi))+1, 1)
    ticks = ns * np.sqrt(np.pi)
    labels = []
    if alt_labels:
        labels = [str(n) for n in ns]
    else:
        for n in ns:
            prefix = str(n)
            match n:
                case -1:
                    prefix = "-"
                case 1:
                    prefix = ""
                case 0:
                    labels.append(r"$0$")
                    continue
            labels.append("$" + prefix + r"\sqrt{\pi}$")
    
    # We remove every second tick label to clean up a bit
    labels = np.array(labels)
    labels[ns % 2 == 1] = ""
    
    return ticks, labels

def plot_single_mode(xs, state):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(xs, np.real(state), "k-", label=r"$\mathrm{Re}(\psi(q))$")
    ax.plot(xs, np.imag(state), "r--", label=r"$\mathrm{Im}(\psi(q))$")
    
    ax.set_xticks(*get_tickmarks(min(xs), max(xs), True))
    ax.set_xlabel(r"$q/\sqrt{\pi}$")
    plt.legend()
    plt.tight_layout()
    
    return fig, ax

def plot_two_mode(x, y, state, projections: bool=False):
    # Setup from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    fig = plt.figure(figsize=(6, 6))
    axs = None
    
    # Add projection plots
    if projections:
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_y = fig.add_subplot(gs[1, 1], sharey=ax)

        dx, dy = (x[-1] - x[0]) / len(x), (y[-1] - y[0]) / len(y)
        y_int = np.einsum("ij,ij -> i", state, state.conj()).real * dy
        x_int = np.einsum("ij,ij -> j", state, state.conj()).real * dx
        
        x_int_span = (min(x_int), max(x_int))
        y_int_span = (min(y_int), max(y_int))
        span = (min(x_int_span+y_int_span), max(x_int_span+y_int_span))
        width = span[1] - span[0]
        lims = (span[0] - width/10, span[1] + width/10)

        ax_x.plot(x, y_int, "k-")
        ax_x.grid(axis="x")
        ax_x.tick_params(axis="x", labelbottom=False)
        ax_x.set_ylim(*lims)

        ax_y.plot(x_int, y, "k-")
        ax_y.grid(axis="y")
        ax_y.tick_params(axis="y", labelleft=False)
        ax_y.set_xlim(*lims)
        axs = [ax, ax_x, ax_y]
    else:
        ax = fig.add_subplot(1, 1, 1)
        axs = ax

    # Plot state
    # ax.contour(*np.meshgrid(x, y, indexing="ij"), np.abs(state), 10, cmap="Kvantify")
    ax.contour(*np.meshgrid(x, y, indexing="ij"), np.abs(state), 10, colors="Black")

    # Fix axes
    ax.set_xticks(*get_tickmarks(min(x), max(x), True))
    ax.set_xlabel(r"$q_1/\sqrt{\pi}$")
    ax.set_yticks(*get_tickmarks(min(y), max(y), True))
    ax.set_ylabel(r"$q_2/\sqrt{\pi}$")
    
    ax.grid()
    
    return fig, axs