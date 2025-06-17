import numpy as np
from constants import *
from costs import *

# --------------------------------------
# 1) FIRST GEAR DYNAMICS
# --------------------------------------
def f_first_gear(v: float, T_M: float) -> float:
    """
    Dynamics for first gear mode
    Args:
        v: longitudinal velocity
        T_M: motor torque
    Returns:
        v_dot: velocity derivative
    """
    return -A1 * v**2 + B1 * T_M - C1 * v - D1

def d_v_f_first_gear(v: float, T_M: float) -> float:
    """
    Derivative of first gear dynamics with respect to velocity
    Args:
        v: longitudinal velocity
        T_M: motor torque
    Returns:
        d_v_f: derivative of f_first_gear with respect to v
    """
    return -2 * A1 * v - C1

# --------------------------------------
# 2) TRANSITION MODE DYNAMICS
# --------------------------------------
def f_transition(omega_S: float, omega_R: float, T_M: float, T_BS: float, T_BR: float) -> tuple[float, float]:
    """
    Dynamics for transition mode
    Args:
        omega_S: sun gear angular velocity
        omega_R: ring gear angular velocity
        T_M: motor torque
        T_BS: sun brake torque
        T_BR: ring brake torque
    Returns:
        omega_S_dot, omega_R_dot: angular velocity derivatives
    """
    combined_omega = omega_S + R2 * omega_R
    
    omega_S_dot = (-A_SS * omega_S + A_SR * omega_R - A_SA * combined_omega**2 
                   + B_SM * T_M + B_SS * T_BS - B_SR * T_BR - D_SL)
    
    omega_R_dot = (A_RS * omega_S - A_RR * omega_R - A_RA * combined_omega**2
                   + B_RM * T_M - B_RS * T_BS + B_RR * T_BR - D_RL)
    
    return omega_S_dot, omega_R_dot

# --------------------------------------
# 3) SECOND GEAR DYNAMICS
# --------------------------------------
def f_second_gear(v: float, T_M: float) -> float:
    """
    Dynamics for second gear mode
    Args:
        v: longitudinal velocity
        T_M: motor torque
    Returns:
        v_dot: velocity derivative
    """
    return -A2 * v**2 + B2 * T_M - C2 * v - D2

# --------------------------------------
# 4) AUGMENTED DYNAMICS (INCLUDING ENERGY STATE)
# --------------------------------------
def f_aug_first_gear(y: tuple[float, float], u: float) -> tuple[float, float]:
    """
    Augmented dynamics for first gear (includes energy accumulation)
    Args:
        y: [v, z] - velocity and accumulated energy
        u: T_M - motor torque
    Returns:
        y_dot: [v_dot, z_dot]
    """
    v, z = y
    T_M = u
    v_dot = f_first_gear(v, T_M)
    z_dot = P_B_first_gear(v, T_M)
    return np.array([v_dot, z_dot])

# --------------------------------------
# 5) AUGMENTED DYNAMICS (TRANSITION MODE)
# --------------------------------------
def f_aug_transition(y: tuple[float, float, float], u: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Augmented dynamics for transition mode
    Args:
        y: [omega_S, omega_R, z] - angular velocities and accumulated energy
        u: [T_M, T_BS, T_BR] - motor and brake torques
    Returns:
        y_dot: [omega_S_dot, omega_R_dot, z_dot]
    """
    omega_S, omega_R, z = y
    T_M, T_BS, T_BR = u
    omega_S_dot, omega_R_dot = f_transition(omega_S, omega_R, T_M, T_BS, T_BR)
    z_dot = P_B_transition(omega_S, omega_R, T_M)
    return np.array([omega_S_dot, omega_R_dot, z_dot])

# --------------------------------------
# 6) AUGMENTED DYNAMICS (SECOND GEAR)
# --------------------------------------
def f_aug_second_gear(y: tuple[float, float], u: float) -> tuple[float, float]:
    """
    Augmented dynamics for second gear
    Args:
        y: [v, z] - velocity and accumulated energy
        u: T_M - motor torque
    Returns:
        y_dot: [v_dot, z_dot]
    """
    v, z = y
    T_M = u
    v_dot = f_second_gear(v, T_M)
    z_dot = P_B_second_gear(v, T_M)
    return np.array([v_dot, z_dot])


