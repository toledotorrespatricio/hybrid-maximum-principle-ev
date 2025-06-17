from constants import *

# --------------------------------------
# 1) MIXED STATE-CONTROL CONSTRAINT FUNCTIONS
# --------------------------------------
def phi_first_gear(v: float, T_M: float) -> float:
    """Mixed constraint for first gear: power constraint"""
    return T_M * (i_fd * GR1 / r_W) * v

def d_v_phi_first_gear(v: float, T_M: float) -> float:
    """Derivative of phi_first_gear with respect to v"""
    return T_M * (i_fd * GR1 / r_W)

def phi_second_gear(v: float, T_M: float) -> float:
    """Mixed constraint for second gear: power constraint"""
    return T_M * (i_fd * GR2 / r_W) * v

def phi_transition(omega_S: float, omega_R: float, T_M: float) -> float:
    """Mixed constraint for transition: power constraint"""
    return T_M * (omega_S + R2 * omega_R) / (1 + R2)

# --------------------------------------
# 2) AUGMENTED MIXED STATE-CONTROL CONSTRAINT FUNCTIONS
# --------------------------------------
def phi_aug_first_gear(y: tuple[float, float], u: float) -> float:
    """
    Augmented mixed constraint for first gear mode
    Args:
        y: [v, z] - velocity and accumulated energy
        u: T_M - motor torque
    Returns:
        phi: mixed constraint value
    """
    v, _ = y  # z is not used in the constraint
    return phi_first_gear(v, u)

def phi_aug_second_gear(y: tuple[float, float], u: float) -> float:
    """
    Augmented mixed constraint for second gear mode
    Args:
        y: [v, z] - velocity and accumulated energy
        u: T_M - motor torque
    Returns:
        phi: mixed constraint value
    """
    v, _ = y  # z is not used in the constraint
    return phi_second_gear(v, u)

def phi_aug_transition(y: tuple[float, float, float], u: tuple[float, float, float]) -> float:
    """
    Augmented mixed constraint for transition mode
    Args:
        y: [omega_S, omega_R, z] - angular velocities and accumulated energy
        u: [T_M, T_BS, T_BR] - motor and brake torques
    Returns:
        phi: mixed constraint value
    """
    omega_S, omega_R, _ = y  # z is not used in the constraint
    T_M, _, _ = u  # Only T_M is used in the constraint
    return phi_transition(omega_S, omega_R, T_M)