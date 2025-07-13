from simulators.dv_simulator.gates import *
from simulators.dv_simulator.states import State

def relabel(circuit: list[Gate], map: dict) -> list[Gate]:
    # Computes a new circuit where qubit indices are mapped to new indices as 
    # i -> map[i]. If an index is not in map, then it is mapped to itself. Note
    # that unused indices in map are silently ignored. This implementation is
    # non-intrusive to the given circuit.
    
    indices = set().union(*[gate.indices for gate in circuit])
    
    full_map = {i: i for i in indices}
    full_map.update(map)

    if len(full_map) != len(set(full_map.values())):
        raise ValueError("Generated mapping is not injective.")
    
    result: list[Gate] = []
    for gate in circuit:
        result.append(gate.copy())
        result[-1].relabel(full_map)
    
    return result

# the returned circuit is guarantied to be only nearest neighbour interactions
# if q2 is neighbour to both q1 and q3
CCZ = [
    CX(2, 1),
    Tdg(1),
    CX(0, 1),
    T(1),

    CX(2, 1),
    Tdg(1),
    CX(0, 1),
    T(1),

    T(2),

    SWAP(1, 2),

    CX(0, 1),
    T(0),
    Tdg(1),
    CX(0, 1),

    SWAP(1, 2),
]

def grover(oracle: list[Gate]) -> list[Gate]:
    return [
        Insert(0, State.ZERO),
        Insert(1, State.ZERO),
        Insert(2, State.ZERO),
        
        H(0),
        H(1),
        H(2),

        *oracle,

        H(0),
        H(1),
        H(2),

        X(0),
        X(1),
        X(2),
        
        *CCZ,

        X(0),
        X(1),
        X(2),

        H(0),
        H(1),
        H(2),
    ]

def int2tag(n: int, N: int=0) -> str:
    return "{0:0{1}b}".format(n, N)

def tag2int(tag: str) -> int:
    return int(tag, 2)

def oracle(tagged: list[int]) -> list[Gate]:
    match sorted(tagged):
        case [3, 6]:
            oracle = [  # tags states |011> and |110>
                CZ(0, 1),
                CZ(1, 2),
            ]
        case [0, 4]:
            oracle = [  # tags states |011> and |110>
                Z(1),
                Z(2),
                CZ(1, 2),
            ]
        case [2, 7]:
            oracle = [  # tags states |000> and |100>
                Z(1),
                CZ(0, 1),
                CZ(1, 2),
            ]
        case _:
            NotImplementedError("Requested oracle not implemented")
    
    return oracle