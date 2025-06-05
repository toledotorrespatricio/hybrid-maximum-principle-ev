import numpy as np

# --------------------------------------
# 1) FUNDAMENTAL CONSTANTS FROM TABLE 1 (Pakniyat & Caines, 2017)
# --------------------------------------

# Vehicle parameters
m = 1000.0          # Vehicle mass [kg]
rho = 1.2           # Air density [kg/m³]
A_f = 2.0           # Frontal area [m²]
C_d = 0.3           # Drag coefficient [-]
C_r = 0.02          # Rolling resistance coefficient [-]
g = 9.81            # Gravitational acceleration [m/s²]

# Final drive
i_fd = 12.0         # Final drive ratio (differential)

# Friction coefficients
C_S = 0.001         # Sun gear viscous friction [Nm·s/rad]
C_R = 0.001         # Ring gear viscous friction [Nm·s/rad]
T_Sf = -0.05        # Sun gear Coulomb friction [Nm]
T_Rf = -0.05        # Ring gear Coulomb friction [Nm]

# Power train parameters
L_TT = 0.1443       # Torque squared coefficient [1/(Nm·s)]
L_T_omega = 1.014   # Torque-omega coupling [-]
L_T = -0.889        # Torque linear coefficient [1/s]
L_omega = 6.884     # Angular velocity coefficient [Nm]

# Inertia parameters [kg·m²]
I_S = 0.0015        # Sun gear inertia
I_R = 0.009         # Ring gear inertia
I_C_in = 0.0014     # Input carrier inertia
I_C_out = 0.1       # Output carrier inertia
I_P_in = 6.08e-6    # Input planet gear inertia
I_P_out = 3.12e-5   # Output planet gear inertia

# Planetary gear masses [kg]
m_P_in = 0.0512     # Input planet gear mass
m_P_out = 0.12113   # Output planet gear mass

# Gear pitch radii [m]
r_S_in = 0.03       # Input sun gear pitch radius
r_S_out = 0.015     # Output sun gear pitch radius
r_R_in = 0.06       # Input ring gear pitch radius
r_R_out = 0.06      # Output ring gear pitch radius
r_P_in = 0.015      # Input planet gear pitch radius
r_P_out = 0.0225    # Output planet gear pitch radius
r_W = 0.3           # Wheel radius
r_C_in = 0.045   # Input carrier radius
# ============================================================================
# MOTOR PARAMETERS
# ============================================================================

T_max = 200.0       # Maximum motor torque [Nm]
P_max = 80000.0     # Maximum motor power [W]
omega_star = 400.0  # Motor speed threshold [rad/s]

# Brake torque limits (estimated - not in Table 1)
T_BS_max = 700.0     # Maximum sun brake torque [Nm]
T_BR_max = 300.0     # Maximum ring brake torque [Nm]

# Additional inertias (estimated - not in Table 1)
I_M = 0.0015        # Motor inertia [kg·m²]
I_W = 0.3           # Single wheel inertia [kg·m²]
I_shaft = 0.01      # Shaft inertia [kg·m²]

# ============================================================================
# PROBLEM PARAMETERS
# ============================================================================

v_target = 27.78    # Target velocity [m/s] (100 km/h)
t_f = 6.3           # Final time for energy optimization [s]

# ============================================================================
# CALCULATED CONSTANTS
# ============================================================================

# Gear ratios (Eq. 26)
R1 = r_R_in / r_S_in    # First stage ratio = 0.06/0.03 = 2.0
R2 = r_R_out / r_S_out  # Second stage ratio = 0.06/0.015 = 4.0

# Transmission gear ratios (Eq. 27, 28)
GR1 = (R2 + 1) / (R1 + 1)           # First gear ratio = 5/3 ≈ 1.667
GR2 = (R2 + 1) * R1 / ((R1 + 1) * R2)  # Second gear ratio = 5*2/(3*4) = 10/12 ≈ 0.833

# Equivalent wheel inertia
J_W = 4 * I_W + I_shaft + i_fd**2 * (I_C_out + 4 * m_P_out * r_W**2)

# Equivalent motor inertia
J_M = I_M + I_C_in + 4 * m_P_in * r_C_in**2  # Note: r_C_in not defined, using approximation

# Total inertia of planetary sets
J_P_in = 4 * I_P_in
J_P_out = 4 * I_P_out

# ============================================================================
# DYNAMICS COEFFICIENTS FOR FIXED GEARS
# ============================================================================

# First gear equivalent mass (ωR = 0)
m_eq_1 = m * (1 + J_W/(m*r_W**2) + 
              (J_M/(R1+1)**2 + I_S + J_P_in/(R1-1)**2 + J_P_out/(R2-1)**2) * 
              i_fd**2 * (R2+1)**2 / (m*r_W**2))

# Second gear equivalent mass (ωS = 0)
m_eq_2 = m * (1 + J_W/(m*r_W**2) + 
              (R1**2*J_M/(R1+1)**2 + I_R + R1**2*J_P_in/(R1-1)**2 + R2**2*J_P_out/(R2-1)**2) * 
              i_fd**2 * (R2+1)**2 / (m*r_W**2*R2**2))

# First gear dynamics coefficients (Eq. 44, 46)
A1 = rho * C_d * A_f / (2 * m_eq_1)
B1 = i_fd * GR1 * T_max / (m_eq_1 * r_W)
C1 = i_fd**2 * (R2 + 1)**2 * C_S / (m_eq_1 * r_W**2)
D1 = m * g * C_r / m_eq_1

# Second gear dynamics coefficients (Eq. 61)
A2 = rho * C_d * A_f / (2 * m_eq_2)
B2 = i_fd * GR2 * T_max / (m_eq_2 * r_W)
C2 = i_fd**2 * (R2 + 1)**2 * C_S / (m_eq_2 * r_W**2)
D2 = m * g * C_r / m_eq_2
# ============================================================================
# TRANSITION MODE DYNAMICS COEFFICIENTS (Appendix B.2)
# ============================================================================

# Inertia matrix components (Eq. B.6)
J_SS = ((m*r_W**2 + J_W)/(i_fd**2*(R2+1)**2) + 
        J_M/(R1+1)**2 + I_S + J_P_in/(R1-1)**2 + J_P_out/(R2-1)**2)

J_RR = ((m*r_W**2 + J_W)*R2**2/(i_fd**2*(R2+1)**2) + 
        J_M*R1**2/(R1+1)**2 + I_R + J_P_in*R1**2/(R1-1)**2 + J_P_out*R2**2/(R2-1)**2)

J_SR = ((m*r_W**2 + J_W)*R2/(i_fd**2*(R2+1)**2) + 
        J_M*R1/(R1+1)**2 - J_P_in*R1/(R1-1)**2 - J_P_out*R2/(R2-1)**2)

# Determinant of inertia matrix
det_J = J_SS * J_RR - J_SR**2

# Transition dynamics coefficients (Eq. B.18, B.19)
# Sun gear dynamics coefficients
A_SS = J_RR * C_S / det_J
A_SR = J_SR * C_R / det_J
A_SA = rho * C_d * A_f * (J_RR - R2*J_SR) * r_W**3 / (2 * det_J * i_fd**3 * (R2+1)**3)

B_SS = J_RR / det_J
B_SR = J_SR / det_J
B_SM = (J_RR - R1*J_SR) / det_J

D_SL = ((J_RR - R2*J_SR) * r_W * m * g * C_r) / (i_fd * (R2+1) * det_J) + \
       (J_SR * T_Rf - J_RR * T_Sf) / det_J

# Ring gear dynamics coefficients
A_RS = J_SR * C_S / det_J
A_RR = J_SS * C_R / det_J
A_RA = rho * C_d * A_f * (R2*J_SS - J_SR) * r_W**3 / (2 * det_J * i_fd**3 * (R2+1)**3)

B_RS = J_SR / det_J
B_RR = J_SS / det_J
B_RM = (R1*J_SS - J_SR) / det_J

D_RL = ((R2*J_SS - J_SR) * r_W * m * g * C_r) / (i_fd * (R2+1) * det_J) + \
       (J_SR * T_Sf - J_SS * T_Rf) / det_J

# ============================================================================
# NUMERICAL PARAMETERS
# ============================================================================

# --------------------------------------
# 1) NUMERICAL PARAMETERS
# --------------------------------------
INTEGRATION_STEPS = 1000  # Number of integration steps
OPTIMIZATION_TOL = 1e-6   # Tolerance for optimization convergence
MAX_ITERATIONS = 1000     # Maximum number of iterations for optimization

# ============================================================================
# PRINT SUMMARY OF KEY CONSTANTS
# ============================================================================