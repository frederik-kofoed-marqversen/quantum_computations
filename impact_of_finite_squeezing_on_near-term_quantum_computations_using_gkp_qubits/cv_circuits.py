from simulators.cv_simulator.gates import *
from simulators.cv_simulator.states import State
from simulators.gkp_simulator.gates import *

def qunaught_error_correction(eps: float):
    return [
        Insert(1, State.QUNAUGHT, gkp_epsilon=eps),
        Insert(2, State.QUNAUGHT, gkp_epsilon=eps),
        BS(2, 1),
        BS(1, 0),
        Mq(0),
        Mp(0),
        # Syndrome correction missing
    ]

def quadrature_correction(eps: float):
    return [
        Insert(1, State.GKP_ZERO, gkp_epsilon=eps),
        CZ(0, 1),
        Mp(1),
        # Syndrome correction missing
    ]

def steane_error_correction(eps: float):
    return [
        *quadrature_correction(eps),
        F(0, dagger=True),
        *quadrature_correction(eps),
        F(0),
    ]

def bell_standard(eps):
    return [
        Insert(0, State.GKP_T, gkp_epsilon=eps),
        Insert(1, State.GKP_PLUS, gkp_epsilon=eps),
        *MBCZ(0, 1, epsilon=eps).compile(),
        # Syndrome correction missing
        F(1),
    ]

def bell_qunaught(eps):
    return [
        Insert(0, State.QUNAUGHT, gkp_epsilon=eps),
        Insert(1, State.QUNAUGHT, gkp_epsilon=eps),
        BS(0, 1),
    ]