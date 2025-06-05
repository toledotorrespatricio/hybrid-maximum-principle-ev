# --------------------------------------
# 1) PURE STATE CONSTRAINT FUNCTIONS
# --------------------------------------
def h_first_gear(v: float) -> float:
    """Pure state constraint for first gear: v >= 0"""
    return -v

def h_second_gear(v):
    """Pure state constraint for second gear: v >= 0"""
    return -v

def h_transition(omega_S: float, omega_R: float) -> float:
    """Pure state constraint for transition: min(omega_S, omega_R) >= 0"""
    return -min(omega_S, omega_R)

# --------------------------------------
# 2) AUGMENTED STATE CONSTRAINT FUNCTIONS
# --------------------------------------
def h_aug_first_gear(y: tuple[float, float]) -> float:
    """
    Augmented state constraint for first gear mode
    Args:
        y: [v, z] - velocity and accumulated energy
    Returns:
        h: state constraint value
    """
    v, _ = y  # z is not used in the constraint
    return h_first_gear(v)

def h_aug_second_gear(y: tuple[float, float]) -> float:
    """
    Augmented state constraint for second gear mode
    Args:
        y: [v, z] - velocity and accumulated energy
    Returns:
        h: state constraint value
    """
    v, _ = y  # z is not used in the constraint
    return h_second_gear(v)

def h_aug_transition(y: tuple[float, float, float]) -> float:
    """
    Augmented state constraint for transition mode
    Args:
        y: [omega_S, omega_R, z] - angular velocities and accumulated energy
    Returns:
        h: state constraint value
    """
    omega_S, omega_R, _ = y  # z is not used in the constraint
    return h_transition(omega_S, omega_R)