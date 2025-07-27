from typing import Callable
from math import log2, ceil
import mpmath
from mpmath import mpf

mpmath.mp.dps = 24


class DepolarisationChannel:
    def __init__(self, error, error_rate: float=1.0):
        pi, px, py, pz = self.to_error_vec(error)
        idleing_matrix = mpmath.matrix([
            [pi, px, py, pz],
            [px, pi, pz, py],
            [py, pz, pi, px],
            [pz, py, px, pi],
        ])

        self.lambdas, self.U = mpmath.eigh(idleing_matrix)
        self.rate = mpf(error_rate)
    
    @staticmethod
    def to_error_vec(error) -> list:
        if isinstance(error, (float, int, mpf)):
            return [1 - error, error / 3, error / 3, error / 3]
        elif len(error) == 3:
            return [1 - sum(error), *error]
        elif len(error) == 4:
            return error
        else:
            raise ValueError("Unknown error type!")

    def apply(self, in_error, time, output_scalar=False):
        in_error_vec = mpmath.matrix(self.to_error_vec(in_error))
        time = mpf(time)
        idleing_amount = [mpmath.power(l, self.rate * time) for l in self.lambdas]

        D = mpmath.diag(idleing_amount)
        U_T = self.U.transpose()
        idleing_matrix = self.U * D * U_T

        out_error = idleing_matrix * in_error_vec

        if output_scalar:
            return sum(out_error[1:])
        else:
            return list(out_error)


def balanced_depolarisation_noise(error: list, p: float, depth: float) -> list:
    # Calculate Mn = M^n by diagonalisation and return Mn * p
    # M = [
    #   [1-p, p/3, p/3, p/3],
    #   [p/3, 1-p, p/3, p/3],
    #   [p/3, p/3, 1-p, p/3],
    #   [p/3, p/3, p/3, 1-p],
    # ]
    
    U = mpmath.matrix([
        [1, 1, 1, 1],
        [1, -1, -1, 1],
        [-1, 1, -1, 1],
        [-1, -1, 1, 1],
    ]) / 2
    
    lambda1_n = (1 - 4/3*mpf(p))**depth
    lambda2_n = 1
    Dn = mpmath.diag([lambda1_n, lambda1_n, lambda1_n, lambda2_n])
    Mn = U * Dn * U.transpose()
    
    result = Mn * mpmath.matrix(error)
    return list(result)


def surface_code_qubits(L: int, total: bool=True, *, rotated: bool=True) -> int | tuple[int, int]:
    # Returns the number of data and ancilla qubits as a tuple in that order
    # (num_data_qubits, num_ancilla_qubits)
    
    # Rotated surface code
    qubits = (L**2, L**2 - 1) if rotated else (L**2 + (L - 1)**2, 2*L*(L - 1))

    return sum(qubits) if total else qubits


# # Numerical values from: fowler_surface_2012
# p_star = mpf("0.57e-2")
# prefactor = mpf("0.03")

# def surface_code_error(L: int, p_local: float) -> float:
#     p_L = surface_code_coeff * (p_local / p_b_star)**(L / 2)
#     return p_L

# def surface_code_size(p_local, p_logical) -> int:
#     L = 2 * log2(p_logical / surface_code_coeff) / log2(p_local / p_b_star)
#     L = ceil(L)
#     return L


def logical_error_rate_bulk_seam(L: int, p_b: float, p_s: float) -> float:
    # Numerical values from supplementary material: ramette_fault-tolerant_2024 (eq. 4)

    # # Analytical values
    # mu_s = 2 * 2 - 1  # 2 * D_s - 1
    # mu_b = 2 * 3 - 1  # 2 * D_b - 1
    # alpha_c = 8.0 * mu_s
    # p_b_star = (2 * mu_b)**-2
    # p_s_star = (2 * mu_s)**-2
    # surface_code_coeff = 0.03  # fowler_surface_2012 (is widely accepted)

    # Numerical values from the supplementary material
    p_b_star = mpf("0.75e-2")
    p_s_star = mpf("10.4e-2")
    alpha_c = mpf("1.4")
    # Values extracted from supplementary figure 2 
    # a, b, Db = 1e-1, 1e-4, -118.833
    # x = lambda Dx: a * (b / a)**(Dx / Db)  # conversion from plot distance to axis value
    a_b = 8e-2
    a_s = 0.15429674683914762  # = x(7.461)
    a_bs = 0.0104242833132694  # fitted
    
    # if p_b > p_b_star or p_s > p_s_star:
    #     return None

    p_star_1s = p_s_star * (1 + alpha_c * p_b * (p_s_star)**0.5 / (1 - (p_b / p_b_star)**0.5))**(-2)
    
    # if p_s > p_star_1s:
    #     return None
    
    exp_s = (p_s / p_s_star) ** (L / 2)
    exp_b = (p_b / p_b_star) ** (L / 2)
    exp_comb = sum((p_s / p_star_1s)**(gs / 2) * (p_b / p_b_star)**((L - gs)/2) for gs in range(1, L+1))
    
    p_logical = a_s * exp_s + a_b * exp_b + a_bs * exp_comb
    return p_logical


# Faster method for no seam cases!
def surface_code_error(L: int, p_local: float) -> float:
    surface_code_coeff = mpf("8e-2")
    p_star = mpf("0.75e-2")
    
    p_L = surface_code_coeff * (p_local / p_star)**(L / 2)
    return p_L


# Faster method for no seam cases!
def surface_code_size(p_local, p_logical) -> int:
    surface_code_coeff = mpf("8e-2")
    p_star = mpf("0.75e-2")

    L = 2 * log2(p_logical / surface_code_coeff) / log2(p_local / p_star)
    L = ceil(L)
    return L


def find_code_size(
    code_error: Callable[[int], float],
    p_target: float,
    args: tuple=(),
    stepsize: int=100,
    always_return: bool=False
) -> tuple[int, float]:
    upper = 1
    p = 1
    while True:
        q = code_error(upper, *args)
        if q < p_target or q > p:
            break
        p = q
        upper += stepsize

    # If a solution exists we have now overstepped it.
    # Ternary search for a reduced interval containing L closest to p_target
    f = lambda L: abs(p_target - code_error(L, *args))
    low = upper - stepsize
    high = upper
    while high - low > 3:
        m1 = low + (high - low) // 3
        m2 = high - (high - low) // 3
        if f(m1) < f(m2):
            high = m2
        else:
            low = m1
    
    # Final brute force check over reduced interval
    Ls = list(range(low, high+1))
    ps = []
    # Return solution
    for L in Ls:
        p = code_error(L, *args)
        if p < p_target:
            return (L, p)
        ps.append(p)
    
    # No solution exists
    if not always_return:
        raise ValueError("No solution exists!")
    else:
        return min(zip(Ls, ps), key=lambda x: x[1])


def surface_code_size_bulk_seam(p_bulk: float, p_seam: float, p_logical: float) -> int:
    return find_code_size(logical_error_rate_bulk_seam, p_logical, args=(p_bulk, p_seam))[0]


def transversal_gate_rate(L: int, r_physical: float, r_bell: float, memory: int) -> float:
    num_data_qubits, num_ancilla_qubits = surface_code_qubits(L, False, rotated=False)
    num_qubits = num_data_qubits + num_ancilla_qubits

    process_rate = r_physical / 5
    process_size = num_data_qubits + num_qubits
    # Number of parallel processes that fit within memory
    n = memory // process_size

    r_prepare = r_bell / num_data_qubits
    r_consumption = process_rate * n
    
    return min(r_prepare, r_consumption)


def lattice_surgery_gate_rate(L: int, r_physical: float, r_bell: float, memory: int) -> float:
    num_data_qubits, num_ancilla_qubits = surface_code_qubits(L, False, rotated=False)
    num_qubits = num_data_qubits + num_ancilla_qubits
    
    num_edge_qubits = L
    num_rounds = L
    
    round_rate = r_physical / 5
    process_size = num_edge_qubits + num_qubits
    process_rate = round_rate / num_rounds
    # Number of parallel processes that fit within memory taking into acount buffer
    n = memory // process_size

    r_prepare = r_bell / (num_edge_qubits * num_rounds)
    r_consumption = n * process_rate

    return min(r_prepare, r_consumption)


"""Find the value x for which f(x)=0 with at most a relative error set by reltol by bisection."""
def find_root_bisection(f, a, b, reltol=mpf('1e-6'), maxiter=1000):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        print(fa, fb)
        raise ValueError("Function must change sign over the interval [a, b].")

    for _ in range(maxiter):
        mid = (a + b) / 2
        fmid = f(mid)

        # Termination based on relative difference in x
        interval = b - a
        if abs(interval / mid) < reltol:
            return mid

        # Bisection step
        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    raise RuntimeError("Maximum iterations exceeded without reaching relative tolerance.")