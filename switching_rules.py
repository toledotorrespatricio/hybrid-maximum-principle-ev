import numpy as np
from constants import *

# Switching condition functions
def switch_first_to_transition(v: float) -> tuple[float, float]:
    """
    Compute initial state for transition mode when switching from first gear
    Args:
        v: velocity at switch time
    Returns:
        [omega_S, omega_R]: initial angular velocities for transition
    """
    omega_S = (i_fd * (1 + R2) / r_W) * v
    omega_R = 0.0
    return np.array([omega_S, omega_R])

def switch_transition_to_second(omega_S: float, omega_R: float) -> float:
    """
    Compute initial velocity for second gear when switching from transition
    Args:
        omega_S, omega_R: angular velocities at switch time
    Returns:
        v: initial velocity for second gear
    """
    v = (r_W / (i_fd * (1 + R2))) * (omega_S + R2 * omega_R)
    return v