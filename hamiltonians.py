from constants import *
from dynamics import *
import numpy as np
from typing import Union

# --------------------------------------
# 1) HAMILTONIAN FUNCTIONS (AUGMENTED FORM, INNER PRODUCT)
# --------------------------------------
def H1(
    q: np.ndarray,  # [q_v, q_z]
    y: np.ndarray,  # [v, z]
    u: Union[float, np.float64]  # T_M
) -> float:
    """
    Hamiltonian for first gear mode (augmented).
    Implements: H1(q, y, u) = <q, f_aug_first_gear(y, u)>
    Args:
        q: np.ndarray, shape (2,) - [q_v, q_z] costate vector
        y: np.ndarray, shape (2,) - [v, z] state vector
        u: float - T_M, control (scalar)
    Returns:
        Hamiltonian value (float)
    """
    return np.dot(q, f_aug_first_gear(y, u))

def H2(
    q: np.ndarray,  # [q_omegaS, q_omegaR, q_z]
    y: np.ndarray,  # [omega_S, omega_R, z]
    u: np.ndarray   # [T_M, T_BS, T_BR]
) -> float:
    """
    Hamiltonian for transition mode (augmented).
    Implements: H2(q, y, u) = <q, f_aug_transition(y, u)>
    Args:
        q: np.ndarray, shape (3,) - [q_omegaS, q_omegaR, q_z] costate vector
        y: np.ndarray, shape (3,) - [omega_S, omega_R, z] state vector
        u: np.ndarray, shape (3,) - [T_M, T_BS, T_BR] control vector
    Returns:
        Hamiltonian value (float)
    """
    return np.dot(q, f_aug_transition(y, u))


def H3(
    q: np.ndarray,  # [q_v, q_z]
    y: np.ndarray,  # [v, z]
    u: Union[float, np.float64]  # T_M
) -> float:
    """
    Hamiltonian for third (second gear) mode (augmented).
    Implements: H3(q, y, u) = <q, f_aug_second_gear(y, u)>
    Args:
        q: np.ndarray, shape (2,) - [q_v, q_z] costate vector
        y: np.ndarray, shape (2,) - [v, z] state vector
        u: float - T_M, control (scalar)
    Returns:
        Hamiltonian value (float)
    """
    return np.dot(q, f_aug_second_gear(y, u))



