from normal_cone import Interval
from constants import T_max, T_BS_max, T_BR_max, P_max
from typing import Sequence, Callable, Union, Optional
from mixed_constraints import phi_aug_first_gear, phi_aug_second_gear, phi_aug_transition
import numpy as np

# --------------------------------------
# BOX CLASS FOR MULTIDIMENSIONAL INTERVALS
# --------------------------------------
class Box:
    def __init__(self, intervals: Sequence[Interval]):
        self.intervals = intervals
    def __contains__(self, values):
        if len(values) != len(self.intervals):
            return False
        return all(v in interval for v, interval in zip(values, self.intervals))
    def __repr__(self):
        return f"Box({self.intervals})"

# --------------------------------------
# CONTROL SETS U_alpha
# --------------------------------------
U_1 = Interval(-T_max, T_max)
U_2 = Box([
    Interval(-T_max, T_max),
    Interval(-T_BS_max, 0),
    Interval(-T_BR_max, 0)
])
U_3 = Interval(-T_max, T_max)

# --------------------------------------
# MIXED CONSTRAINT SETS Phi_alpha
# --------------------------------------
Phi_1 = Interval(-P_max, P_max)
Phi_2 = Interval(-P_max, P_max)
Phi_3 = Interval(-P_max, P_max)

# --------------------------------------
# HELPER: INTERSECTION OF INTERVALS
# --------------------------------------
def interval_intersection(a: Interval, b: Interval) -> Optional[Interval]:
    lower = max(a.lower, b.lower)
    upper = min(a.upper, b.upper)
    if lower > upper:
        return None
    return Interval(lower, upper)

# --------------------------------------
# ADMISSIBLE CONTROL SETS Sigma_k(y)
# --------------------------------------
def Sigma_1(y) -> Optional[Interval]:
    """
    Returns the admissible control interval for mode 1 and state y.
    Efficient: computes the intersection of U_1 and the set induced by the mixed constraint.
    """
    v, z = y
    # phi_aug_first_gear(y, u) = K * u * v, K = (i_fd * GR1 / r_W)
    # Want: phi in Phi_1 => -P_max <= K * u * v <= P_max
    # If v == 0, any u in U_1 is admissible
    if np.isclose(v, 0.0):
        return U_1
    # Solve for u: -P_max <= K * u * v <= P_max
    K = (T_max * 0 + 1)  # Dummy, will be replaced below
    from constants import i_fd, GR1, r_W
    K = (i_fd * GR1 / r_W) * v
    tm1 = -P_max / K
    tm2 =  P_max / K
    lower_phi = min(tm1, tm2)
    upper_phi = max(tm1, tm2)
    phi_interval = Interval(lower_phi, upper_phi)
    return interval_intersection(U_1, phi_interval)

def Sigma_3(y) -> Optional[Interval]:
    """
    Returns the admissible control interval for mode 3 and state y.
    Efficient: computes the intersection of U_3 and the set induced by the mixed constraint.
    """
    v, z = y
    # phi_aug_second_gear(y, u) = K * u * v, K = (i_fd * GR2 / r_W)
    if np.isclose(v, 0.0):
        return U_3
    from constants import i_fd, GR2, r_W
    K = (i_fd * GR2 / r_W) * v
    tm1 = -P_max / K
    tm2 =  P_max / K
    lower_phi = min(tm1, tm2)
    upper_phi = max(tm1, tm2)
    phi_interval = Interval(lower_phi, upper_phi)
    return interval_intersection(U_3, phi_interval)

def Sigma_2(y, num_points: int = 30):
    """
    Returns a list of admissible controls (T_M, T_BS, T_BR) for mode 2 and state y.
    Efficient grid search over U_2, filtered by the mixed constraint.
    Args:
        y: state vector [omega_S, omega_R, z]
        num_points: number of grid points per dimension (default 30)
    Returns:
        List of tuples (T_M, T_BS, T_BR) that are admissible
    """
    intervals = U_2.intervals
    T_M_grid = np.linspace(intervals[0].lower, intervals[0].upper, num_points)
    T_BS_grid = np.linspace(intervals[1].lower, intervals[1].upper, num_points)
    T_BR_grid = np.linspace(intervals[2].lower, intervals[2].upper, num_points)
    admissible_controls = []
    for T_M in T_M_grid:
        for T_BS in T_BS_grid:
            for T_BR in T_BR_grid:
                u = (T_M, T_BS, T_BR)
                if phi_aug_transition(y, u) in Phi_2:
                    admissible_controls.append(u)
    return admissible_controls

# --------------------------------------
# SWITCHING SETS (DETERMINISTIC JUMPS)
# --------------------------------------
def S_1_2(v: float, z: float):
    """
    Switching set from first gear to transition.
    Given (v, z), returns ((omega_S, omega_R), z_star) according to the switching rule.
    """
    from constants import i_fd, R2, r_W
    omega_S = (i_fd * (1 + R2) / r_W) * v
    omega_R = 0.0
    z_star = z
    return (omega_S, omega_R, z_star)

def S_2_3(omega_S: float, omega_R: float, z: float):
    """
    Switching set from transition to second gear.
    Given (omega_S, omega_R, z), returns (v, z_star) according to the switching rule.
    """
    from constants import i_fd, R2, r_W
    v = (r_W / (i_fd * (1 + R2))) * (omega_S + R2 * omega_R)
    z_star = z
    return (v, z_star)
