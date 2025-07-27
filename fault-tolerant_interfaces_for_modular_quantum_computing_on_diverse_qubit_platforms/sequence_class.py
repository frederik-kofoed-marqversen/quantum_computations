from abc import ABC, abstractmethod
import json
import mpmath
from mpmath import mpf, binomial, inf
mpmath.mp.dps = 24
from ConstantRateDistillation.Distillation_functions import ED_n_1_n
from utils import surface_code_qubits, surface_code_error, balanced_depolarisation_noise


def scalar_error(p):
    if isinstance(p, mpf):
        return p
    elif isinstance(p, list) and len(p) >= 4:
        return mpf(p[1] + p[2] + p[3])
    else:
        raise ValueError("Invalid input. Expected an mpf number or a list with at least four elements.")


class Stage(ABC):
    _subclass_registry = {}
    
    def __init__(self, code, L, p_L, p_local):
        self.n, self.k, self.d = code
        self.L: int = L
        self.p_L: float = p_L
        self.p_local: float = p_local
        self.qubit_size: int = surface_code_qubits(L)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Stage._subclass_registry[cls.__name__] = cls

    def _serialisable_args(self) -> list:
        return [repr(arg) if isinstance(arg, mpf) else arg for arg in self.args()]
    
    @staticmethod
    def _from_serialised_args(args: list) -> list:
        return [mpf(arg[5:-2]) if isinstance(arg, str) and arg[:3] == "mpf" else arg for arg in args]

    def serialise(self) -> str:
        return json.dumps({
            "type": self.__class__.__name__,
            "args": self._serialisable_args(),
        })

    @classmethod
    def from_serialised(cls, json_str: str) -> "Stage":
        data = json.loads(json_str)
        stage_cls = cls._subclass_registry.get(data["type"])
        if stage_cls is None:
            raise ValueError(f"Unknown stage type: {data['type']}")
        args = cls._from_serialised_args(data["args"])
        return stage_cls(*args)

    @abstractmethod
    def __str__(self) -> str: ...
    @abstractmethod
    def args(self) -> list: ...
    @abstractmethod
    def get_logical_depth(self) -> int: ...
    @abstractmethod
    def get_physical_depth(self) -> int: ...
    @abstractmethod
    def compute_error_metrics(self, in_error: float | list[float]) -> tuple[float | list[float], float]: ...


class QuantumStage(Stage):
    def __str__(self):
        return f"[{[self.n, self.k, self.d]}]"
    
    def args(self): return [(self.n, self.k, self.d), self.L, self.p_L, self.p_local]
    def get_logical_depth(self): return 3*self.n - 2 - self.k
    def get_physical_depth(self): return self.get_logical_depth() * 5
    def compute_error_metrics(self, in_error):
        in_error = scalar_error(in_error)  # make in_error scalar if Pauli error vector
        q = (1 - in_error) * ((1 - self.p_L)**self.get_logical_depth())
        bin_sum = sum(binomial(self.n, i) * (1 - q)**i * q**(self.n - i) for i in range(self.d))
        qn = q**self.n
        out_error = (1 - bin_sum) / qn
        p_fail = 1 - qn
        return out_error, p_fail


class ClassicalStage(Stage):
    def __init__(self, code, basis, L, p_L, p_local):
        self.basis = basis
        super().__init__(code, L, p_L, p_local)
        if self.n != self.d:
            raise NotImplementedError("Only [n, 1, n] classical codes are implemented.")
        
    def __str__(self):
        return f"{[self.n, self.k, self.d]}_{self.basis}"

    def args(self): return [(self.n, self.k, self.d), self.basis, self.L, self.p_L, self.p_local]
    def get_logical_depth(self): return 3*self.n - 2 - self.k
    def get_physical_depth(self): return self.get_logical_depth() * 5
    def compute_error_metrics(self, in_error):
        rate, out_error, _ = ED_n_1_n(self.n, in_error=in_error, basis=self.basis)
        p_fail = 1 - self.n * rate
        out_error = balanced_depolarisation_noise(out_error, self.p_L, self.get_logical_depth())
        return out_error, p_fail


class InitStage(Stage):
    def __init__(self, error, L, p_local, p_L=None):
        p_L = surface_code_error(L, p_local) if p_L is None else p_L
        super().__init__((1, 1, 0), L, p_L, p_local)
        self.error = error
    
    def __str__(self):
        return f"Initialisation"
    
    def args(self): return [self.error, self.L, self.p_local]
    def get_logical_depth(self): return 0
    def get_physical_depth(self): return 0
    def compute_error_metrics(self, _in_error):
        return self.error, 0.0


class InjectionStage(Stage):
    def __init__(self, L, p_local):
        if L != 3:
            raise NotImplementedError(f"Injection into code size {L} not implemented.")
        if str(p_local) != "0.001":
            raise NotImplementedError("Injection only implemented for p_local = 0.1%")
        p_L = surface_code_error(L, p_local)
        super().__init__((1, 1, 0), L, p_L, p_local)
        self.p_fail = 1 - (1 - mpf("8e-2"))**2
    
    def __str__(self):
        return f"Injection"
    
    def args(self): return [self.L, self.p_local]
    def get_logical_depth(self): return 0
    def get_physical_depth(self): return 2 * 5  # two rounds of syndrome extraction
    def compute_error_metrics(self, in_error):
        match str(in_error):
            case "0.01":
                return mpf("1.25e-2"), self.p_fail
            case "0.05":
                return mpf("5.2e-2"), self.p_fail
            case _:
                raise NotImplementedError("Injection only implemented for 1% and 5% input errors")


class GrowStage(Stage):
    def __init__(self, L_out, L_in, p_L_in, p_local):
        self.L_in = L_in
        self.p_L_in = p_L_in
        p_L_out = surface_code_error(L_out, p_local)
        super().__init__((1, 1, 0), L_out, p_L_out, p_local)
    
    def __str__(self):
        return f"Growing"
    
    def args(self): return [self.L, self.L_in, self.p_L_in, self.p_local]
    def get_logical_depth(self): return 2
    def get_physical_depth(self): return self.get_logical_depth() * self.L_in * 4
    def compute_error_metrics(self, in_error):
        depth = self.get_logical_depth()
        p_L = self.p_L_in
        p_fail = 0.0
        if isinstance(in_error, list):
            p_out = balanced_depolarisation_noise(in_error, p_L, depth)
        else:
            q = (1 - in_error) * ((1 - p_L)**depth)
            p_out = 1 - q

        return p_out, p_fail


class LogicalDistillationSequence:
    def __init__(self, init_stage: InitStage):
        self.stages: list[Stage] = []
        self.stage_p_fail = []
        self.stage_p_out = []
        self.min_memory_req: int = 0
        self.encoding_rate: float = 1
        self.M: float = 0
        self.M_idle: float = 0
        self.K: int = 1

        # Add initial stage
        self.stages.append(init_stage)
        self.stage_p_fail.append(mpf(0.0))
        self.stage_p_out.append(init_stage.error)

    def __str__(self):
        lines = ["Distillation stages:"]
        for stage, p_out in zip(self.stages, self.stage_p_out):
            line = (
                f"{str(stage):<15}: "
                f"L={stage.L}, "
                f"p_L={float(stage.p_L):.3e}, "
                f"p_out={float(scalar_error(p_out)):.3e}"
            )
            lines.append(line)
        
        summary = (
            "Summary: "
            f"logical error rate={float(self.p_out):.3e}, "
            f"memory requirement={self.min_memory_req}, "
            f"encoding rate={float(self.encoding_rate):.3e}"
        )
        lines.append(summary)

        return "\n".join(lines)

    def serialise(self) -> str:
        return json.dumps([stage.serialise() for stage in self.stages])

    @staticmethod
    def deserialise(data_str: str) -> 'LogicalDistillationSequence':
        strs = iter(json.loads(data_str))
        seq = LogicalDistillationSequence(Stage.from_serialised(next(strs)))
        for serialised_stage in strs:
            stage = Stage.from_serialised(serialised_stage)
            seq.add_stage(stage)
        return seq

    def add_stage(self, stage: Stage):
        n, k = stage.n, stage.k
        p_out, p_fail = stage.compute_error_metrics(self.stage_p_out[-1])

        T = stage.get_physical_depth()
        K = self.K
        E = self.encoding_rate
        size = stage.qubit_size

        min_mem = self.min_memory_req
        dM = (size - self.qubit_size) * n * K

        # No reading of values from this point (for safety)
        self.stages.append(stage)
        self.stage_p_fail.append(p_fail)
        self.stage_p_out.append(p_out)
        self.min_memory_req = max(n*K*size, (n-1)*K*size + min_mem + dM)
        self.encoding_rate *= (1 - p_fail) * k / n
        self.M += T * E * K * size
        self.M_idle += size * K * (n-1) / 2
        self.K *= k

    def shallow_copy(self) -> 'LogicalDistillationSequence':
        copy = LogicalDistillationSequence.__new__(LogicalDistillationSequence)
        
        copy.stages = self.stages.copy()
        copy.stage_p_fail = self.stage_p_fail.copy()
        copy.stage_p_out = self.stage_p_out.copy()
        copy.min_memory_req = self.min_memory_req
        copy.encoding_rate = self.encoding_rate
        copy.M = self.M
        copy.M_idle = self.M_idle
        copy.K = self.K
        return copy

    @property
    def p_out(self): return scalar_error(self.stage_p_out[-1])

    @property
    def p_L(self): return self.stages[-1].p_L

    @property
    def L(self): return self.stages[-1].L

    @property
    def qubit_size(self): return self.stages[-1].qubit_size

    def input_rate_cap(self, allocated_memory: int, local_gate_rate: float = 1.0) -> float:
        cap = local_gate_rate * (allocated_memory - self.M_idle) / self.M
        return max(0.0, cap)

    def distillation_rate(self, allocated_memory: int, max_input_rate: float = inf, local_gate_rate: float = 1.0) -> float:
        if allocated_memory < self.min_memory_req:
            return 0.0
        if max_input_rate == 0.0:
            return self.encoding_rate
        input_rate = self.input_rate_cap(allocated_memory, local_gate_rate)
        return min(max_input_rate, input_rate) * self.encoding_rate
