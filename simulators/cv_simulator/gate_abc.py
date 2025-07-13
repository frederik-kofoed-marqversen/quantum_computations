from abc import ABC, abstractmethod
from typing import Any
from .mps import (MPS, SVD_OPTIONS)
import logging
logger = logging.getLogger(__name__)

"""
When defining new gates the priority order of Mixins is before that of any Gate types: (...Mixin, Gate)
"""

# Number of digits to print with gates
REPR_DIGITS = 5


class MeasurementResult:
    def __init__(self, result: float, probability: float):
        self.result: float = result
        self.probability: float = probability
    
    def __repr__(self):
        return str(self.result)


class Gate(ABC):
    """Abstract base class for CV quantum gates."""

    def __init__(self, arg: Any=None, dagger: bool=False, **kwargs):
        """
        Initialize the gate. Unexpected keyword arguments are logged.
        
        :param arg: The argument for parametrised gates.
        :param dagger: Whether to initialize the adjoint gate.
        :param **kwargs: Include options for svd trunctation.
        """
        self.arg = arg
        self.dagger = dagger
        self.svd_options = {key: kwargs.pop(key) for key in SVD_OPTIONS if key in kwargs}

        if kwargs:
            logger.warning(f"{type(self).__name__} recieved unexpected keyword arguments: {kwargs.keys()}")

    def __repr__(self):
        arg = self.arg
        arg = round(arg, REPR_DIGITS) if isinstance(arg, float) else arg
        return type(self).__name__ + (f"({arg})" if arg is not None else "") + ("^†" if self.dagger else "")

    @abstractmethod
    def apply(self, mps: MPS, **kwargs) -> None | MeasurementResult:
        """
        Apply the gate logic to the given mps modifying it in-place and
        return the result if Gate is a measurement.
        
        :param mps: The MPS state that Gate is applied to.
        :param kwargs: Relevant arguments for the gate. Any non-relevant keyword arguments passed to Gate should be ignored silently.
        """
        pass


class SingleModeGate(Gate):
    """Abstract base class for single-mode gates."""
    
    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(index, int):
            raise ValueError(f"{type(self).__name__} requires a single integer index.")
        self.index = index
    
    def __repr__(self):
        return super().__repr__() + f"_{self.index}"


class Measurement(SingleModeGate):
    def __init__(self, index, result: float=None, **kwargs):
        if kwargs.pop("dagger", None):
            logger.info(type(self).__name__ + "gates ignores adjoint/dagger.")
        super().__init__(index, **kwargs)
        self.result: float = result
    
    def __repr__(self):
        return super().__repr__() + (f" = {round(self.result, REPR_DIGITS)}" if self.result else "")
    
    @abstractmethod
    def apply(self, mps: MPS, **kwargs) -> MeasurementResult:
        pass


class TwoModeGate(Gate):
    """Abstract base class for two-mode gates."""
    
    def __init__(self, index1: int, index2: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(index1, int) or not isinstance(index2, int):
            raise ValueError(f"{type(self).__name__} requires exactly two indices.")
        if abs(index1 - index2) != 1:
            raise ValueError(f"{type(self).__name__} can only be applied to neighbours, but indices: {(index1, index2)} were given.")
        self.index1, self.index2 = index1, index2
        self.left_index, self.right_index = sorted([index1, index2])
    
    def __repr__(self):
        return super().__repr__() + f"_{self.index1},{self.index2}"