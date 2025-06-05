import numpy as np
from typing import Union, Tuple, Optional, List

# --------------------------------------
# 1) INTERVAL TYPE DEFINITIONS
# --------------------------------------
class Interval:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
    
    def __contains__(self, value: float) -> bool:
        return self.lower <= value <= self.upper
    
    def __repr__(self) -> str:
        return f"Interval({self.lower}, {self.upper})"

NormalCone = Union[Interval, None]

def normal_cone_interval(x: float, a: float, b: float) -> NormalCone:
    """
    Compute the normal cone to interval [a, b] at point x
    Args:
        x: point in interval
        a, b: interval bounds
    Returns:
        Normal cone at x (as a set representation)
    """
    if x < a or x > b:
        return None  # x not in interval
    elif a < x < b:
        return Interval(0.0, 0.0)  # Interior point
    elif x == a:
        return Interval(-np.inf, 0.0)  # Left boundary
    elif x == b:
        return Interval(0.0, np.inf)   # Right boundary

def normal_cone_box(x: np.ndarray, bounds: List[Tuple[float, float]]) -> Optional[List[NormalCone]]:
    """
    Compute the normal cone to a box constraint at point x
    Args:
        x: point in R^n
        bounds: list of (lower, upper) bounds for each dimension
    Returns:
        Normal cone at x (component-wise)
    """
    n = len(x)
    cone = []
    
    for i in range(n):
        a, b = bounds[i]
        if x[i] < a or x[i] > b:
            return None  # x not in box
        elif a < x[i] < b:
            cone.append(Interval(0.0, 0.0))  # Interior in this dimension
        elif x[i] == a:
            cone.append(Interval(-np.inf, 0.0))  # At lower bound
        elif x[i] == b:
            cone.append(Interval(0.0, np.inf))   # At upper bound
    
    return cone

def project_on_interval(x: float, a: float, b: float) -> float:
    """Project point x onto interval [a, b]"""
    return np.clip(x, a, b)

def project_on_box(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Project point x onto box defined by bounds"""
    projected = np.zeros_like(x)
    for i in range(len(x)):
        projected[i] = project_on_interval(x[i], bounds[i][0], bounds[i][1])
    return projected



