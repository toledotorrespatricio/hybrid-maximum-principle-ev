from constants import *

# --------------------------------------
# 1) POWER/ENERGY CALCULATION FUNCTIONS
# --------------------------------------
def P_B_first_gear(v: float, T_M: float) -> float:
    """Battery power for first gear"""
    omega_equiv = (i_fd * GR1 / r_W) * v
    return L_TT * T_M**2 + L_T * T_M + (L_T_omega * T_M + L_omega) * omega_equiv
    
def P_B_second_gear(v: float, T_M: float) -> float:
    """Battery power for second gear"""
    omega_equiv = (i_fd * GR2 / r_W) * v
    return L_TT * T_M**2 + L_T * T_M + (L_T_omega * T_M + L_omega) * omega_equiv

def P_B_transition(omega_S: float, omega_R: float, T_M: float) -> float:
    """Battery power for transition mode"""
    omega_avg = (omega_S + R2 * omega_R) / (1 + R2)
    return L_TT * T_M**2 + L_T * T_M + (L_T_omega * T_M + L_omega) * omega_avg