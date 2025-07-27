import numpy as np
import mpmath
import pickle
import os
mpmath.mp.dps = 80 
# getcontext().prec = 70  


# Load all the final_prob_dict dictionaries into a dictionary
dir_path = os.path.dirname(os.path.realpath(__file__))
max_rep_code = 12
n_values = np.linspace(2,max_rep_code,max_rep_code-1, dtype=int)
final_prob_dicts = {}
for n in n_values:
    with open(f'{dir_path}/code_data/repetition_code_prob_dict__n_{n}.pkl', 'rb') as file:
        final_prob_dicts[n] = pickle.load(file)


def ED_C_n_1_n(n, p, printing=False): # [n,1,n] repetition code
    final_prob_dict = final_prob_dicts[n]
    
    pI = mpmath.mpf(p[0])
    pX = mpmath.mpf(p[1])
    pZ = mpmath.mpf(p[2])
    pY = mpmath.mpf(p[3])
    
    # Here put the values of pI, pX, pZ, pY so that LpI, LpX, LpZ, LpY will be numbers and not functions:
    LpI_expr = final_prob_dict['IL']
    LpX_expr = final_prob_dict['XL']
    LpZ_expr = final_prob_dict['ZL']
    LpY_expr = final_prob_dict['YL']
    
    # Substitute the values of pI, pX, pZ, pY into the expressions
    LpI = LpI_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
    LpX = LpX_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
    LpZ = LpZ_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
    LpY = LpY_expr.subs({'pI': pI, 'pX': pX, 'pZ': pZ, 'pY': pY})
        
    norm = LpI + LpX + LpZ + LpY
    p_reject = mpmath.mpf(1) - norm # rejection probability
    rate = (mpmath.mpf(1) / mpmath.mpf(n)) * (mpmath.mpf(1) - p_reject)
    if printing:
        print(f"probability of success in [2,1,2] step = {1-float(p_reject):.2e}") # for test - remove!
    return rate, [LpI/norm, LpX/norm, LpZ/norm, LpY/norm]


def depolarizing(p): # from a scalar p to a vector pI, pX, pZ, pY
    if isinstance(p, mpmath.mpf):
        return [mpmath.mpf(1) - p, p / mpmath.mpf(3), p / mpmath.mpf(3), p / mpmath.mpf(3)]
    elif isinstance(p, list):
        if len(p) == 1:
            return [mpmath.mpf(1) - p[0], p[0] / mpmath.mpf(3), p[0] / mpmath.mpf(3), p[0] / mpmath.mpf(3)]
        elif len(p) > 1:
            return p
    raise ValueError("Invalid input. Expected an mpf number or a list.")


def hadamard(p): # I,X,Z,Y --> I,Z,X,Y
    return [p[0], p[2], p[1], p[3]] 


def s_mat(p): # I,X,Z,Y --> I,Y,Z,X
    # we use HSH to take (I,X,Z,Y)->(I,X,Y,Z)
    return [p[0], p[3], p[2], p[1]] 


def ED_n_1_n(n, in_error, basis = 'Z', printing=False): # Classical repetition codes, for different bases. Out error is a vector
    # Change basis:
    if basis == 'X':
        in_error = hadamard(depolarizing(in_error))
    elif basis == 'Y':
        in_error = hadamard(s_mat(hadamard(depolarizing(in_error))))
    
    # Repetition code in Z basis:
    eff_rate, out_error = ED_C_n_1_n(n, depolarizing(in_error), printing=printing)

    # Change basis again:
    if basis == 'X':
        out_error = hadamard(out_error)
    elif basis == 'Y':
        out_error = hadamard(s_mat(hadamard(out_error)))
        
    out_qubits = 1
    return eff_rate, out_error, out_qubits