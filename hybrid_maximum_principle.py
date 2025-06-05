import numpy as np
from scipy.integrate import solve_ivp
from sets import Sigma_1, Sigma_2, Sigma_3, S_1_2, S_2_3, Phi_1, Phi_2, Phi_3
from hamiltonians import H1, H2, H3
from dynamics import f_aug_first_gear, f_aug_second_gear, f_aug_transition
from constants import t_f, i_fd, GR1, GR2, r_W, A1, B1, C1, A2, B2, C2, R2, A_SS, A_SR, A_SA, A_RS, A_RR, A_RA

from plot_hybrid_ocp import plot_hybrid_ocp_solution
from mixed_constraints import phi_aug_first_gear, phi_aug_second_gear, phi_aug_transition
from scipy.optimize import minimize

# Helper: maximization of Hamiltonian over Sigma_k(y)
def maximize_H(H_fun, Sigma_fun, q, y, t, num_points=10):
    # Debug print for performance insight
    print(f"maximize_H called at t={t:.3f}")
    Sigma = Sigma_fun(y)
    if Sigma is None:
        return 0.0  # No admissible control
    if hasattr(Sigma, 'lower') and hasattr(Sigma, 'upper'):
        u_grid = np.linspace(Sigma.lower, Sigma.upper, num_points)
        H_vals = [H_fun(q, y, u) for u in u_grid]
        idx = np.argmax(H_vals)
        return u_grid[idx]
    admissible_controls = Sigma_fun(y, num_points=30)
    if not admissible_controls:
        return np.zeros(3)
    H_vals = [H_fun(q, y, u) for u in admissible_controls]
    idx = np.argmax(H_vals)
    return admissible_controls[idx]

# --- Matriz auxiliar para transición ---
def A_transition(omega_S, omega_R):
    """
    Returns the A(t) matrix for the transition mode.
    """
    a11 = -A_SS - 2*A_SA*omega_S - A_SA*R2*omega_R
    a12 = A_SR - 2*A_SA*R2*omega_S - A_SA*R2**2*omega_R
    a21 = A_RS - 2*A_RA*omega_S - A_RA*R2*omega_R
    a22 = -A_RR - 2*A_RA*R2*omega_S - A_RA*R2**2*omega_R
    return np.array([[a11, a12], [a21, a22]])

# --- Gamma function for mixed constraint ---
def gamma_from_constraint(phi, Phi, value=1.0):
    """
    Returns gamma and active flag for the mixed constraint.
    phi: value of the mixed constraint
    Phi: Interval object
    value: constant to use at the boundary (default 1.0)
    Returns: gamma, active (bool)
    """
    eps = 1e-8
    if Phi.lower + eps < phi < Phi.upper - eps:
        return 0.0, False
    elif np.isclose(phi, Phi.lower, atol=eps):
        return -value, True
    elif np.isclose(phi, Phi.upper, atol=eps):
        return value, True
    else:
        # Out of bounds (should not happen if control is admissible)
        return 0.0, False

# --- Adjoint equations for each mode ---
def dH1_dy(q, y, u, gamma):
    q_v, q_z = q
    v, z = y
    T_M = u
    dq_v = q_v * (-2*A1*v - C1) - gamma * (i_fd*GR1/r_W) * v
    dq_z = 0
    return np.array([dq_v, dq_z])

def dH3_dy(q, y, u, gamma):
    q_v, q_z = q
    v, z = y
    T_M = u
    dq_v = q_v * (-2*A2*v - C2) - gamma * (i_fd*GR2/r_W) * v
    dq_z = 0
    return np.array([dq_v, dq_z])

def dH2_dy(q, y, u, gamma):
    # q = [q_omegaS, q_omegaR, q_z], y = [omega_S, omega_R, z], u = [T_M, T_BS, T_BR]
    omega_S, omega_R, z = y
    T_M, T_BS, T_BR = u
    # Vector for T_M in the constraint
    v_TM = np.array([T_M/(1+R2), T_M*R2/(1+R2)])
    A = A_transition(omega_S, omega_R)
    dq_omega = A @ v_TM * q[:2]  # Only first two components
    dq_z = 0
    return np.array([dq_omega[0], dq_omega[1], dq_z])


def solve_hybrid_ocp_3mode(
    y0, qf, t_switch, t_final=t_f, num_points=100, gamma_value=1.0, rtol=1e-3, atol=1e-6
):
    print(f'>> solve_hybrid_ocp_3mode: t_switch={t_switch}')
    t1, t2 = t_switch
    # Forward pass: integrate state
    print('Integrando dinámica hacia adelante (modo 1)...')
    q1_interp = lambda t: np.array([0.0, 0.0])
    q2_interp = lambda t: np.array([0.0, 0.0, 0.0])
    q3_interp = lambda t: np.array([0.0, 0.0])
    def dyn1(t, y):
        u = maximize_H(H1, Sigma_1, q1_interp(t), y, t, num_points)
        return f_aug_first_gear(y, u)
    sol1 = solve_ivp(dyn1, [0, t1], y0, dense_output=True, rtol=rtol, atol=atol, max_step=0.1)
    y1 = sol1.y.T
    t1_grid = sol1.t
    print('Integrando dinámica hacia adelante (modo 2)...')
    y2_0 = S_1_2(y1[-1,0], y1[-1,1])
    def dyn2(t, y):
        u = maximize_H(H2, Sigma_2, q2_interp(t), y, t, num_points)
        return f_aug_transition(y, u)
    sol2 = solve_ivp(dyn2, [t1, t2], y2_0, dense_output=True, rtol=rtol, atol=atol, max_step=0.1)
    y2 = sol2.y.T
    t2_grid = sol2.t
    print('Integrando dinámica hacia adelante (modo 3)...')
    y3_0 = S_2_3(y2[-1,0], y2[-1,1], y2[-1,2])
    def dyn3(t, y):
        u = maximize_H(H3, Sigma_3, q3_interp(t), y, t, num_points)
        return f_aug_second_gear(y, u)
    sol3 = solve_ivp(dyn3, [t2, t_final], y3_0, dense_output=True, rtol=rtol, atol=atol, max_step=0.1)
    y3 = sol3.y.T
    t3_grid = sol3.t
    # Backward pass: integrate adjoint
    # 3. Gear 2 adjoint
    def adj3(t, q, y, u, gamma):
        return -dH3_dy(q, y, u, gamma)
    q3 = np.zeros_like(y3)
    gamma3 = np.zeros(len(t3_grid))
    active3 = np.zeros(len(t3_grid), dtype=bool)
    q3[-1] = qf
    for i in range(len(t3_grid)-2, -1, -1):
        dt = t3_grid[i+1] - t3_grid[i]
        u = maximize_H(H3, Sigma_3, q3[i+1], y3[i+1], t3_grid[i+1], num_points)
        phi = phi_aug_second_gear(y3[i+1], u)
        gamma, active = gamma_from_constraint(phi, Phi_3, gamma_value)
        dq = adj3(t3_grid[i+1], q3[i+1], y3[i+1], u, gamma)
        q3[i] = q3[i+1] - dt * dq
        gamma3[i] = gamma
        active3[i] = active
    # 2. Transition adjoint
    def adj2(t, q, y, u, gamma):
        return -dH2_dy(q, y, u, gamma)
    q2 = np.zeros_like(y2)
    gamma2 = np.zeros(len(t2_grid))
    active2 = np.zeros(len(t2_grid), dtype=bool)
    for i in range(len(t2_grid)-2, -1, -1):
        dt = t2_grid[i+1] - t2_grid[i]
        u = maximize_H(H2, Sigma_2, q2[i+1], y2[i+1], t2_grid[i+1], num_points)
        phi = phi_aug_transition(y2[i+1], u)
        gamma, active = gamma_from_constraint(phi, Phi_2, gamma_value)
        dq = adj2(t2_grid[i+1], q2[i+1], y2[i+1], u, gamma)
        q2[i] = q2[i+1] - dt * dq
        gamma2[i] = gamma
        active2[i] = active
    # 1. Gear 1 adjoint
    def adj1(t, q, y, u, gamma):
        return -dH1_dy(q, y, u, gamma)
    q1 = np.zeros_like(y1)
    gamma1 = np.zeros(len(t1_grid))
    active1 = np.zeros(len(t1_grid), dtype=bool)
    for i in range(len(t1_grid)-2, -1, -1):
        dt = t1_grid[i+1] - t1_grid[i]
        u = maximize_H(H1, Sigma_1, q1[i+1], y1[i+1], t1_grid[i+1], num_points)
        phi = phi_aug_first_gear(y1[i+1], u)
        gamma, active = gamma_from_constraint(phi, Phi_1, gamma_value)
        dq = adj1(t1_grid[i+1], q1[i+1], y1[i+1], u, gamma)
        q1[i] = q1[i+1] - dt * dq
        gamma1[i] = gamma
        active1[i] = active
    # Control and Hamiltonian
    u1 = np.array([maximize_H(H1, Sigma_1, q1[i], y1[i], t1_grid[i], num_points) for i in range(len(t1_grid))])
    u2 = np.array([maximize_H(H2, Sigma_2, q2[i], y2[i], t2_grid[i], num_points) for i in range(len(t2_grid))])
    u3 = np.array([maximize_H(H3, Sigma_3, q3[i], y3[i], t3_grid[i], num_points) for i in range(len(t3_grid))])
    H1_arr = np.array([H1(q1[i], y1[i], u1[i]) for i in range(len(t1_grid))])
    H2_arr = np.array([H2(q2[i], y2[i], u2[i]) for i in range(len(t2_grid))])
    H3_arr = np.array([H3(q3[i], y3[i], u3[i]) for i in range(len(t3_grid))])
    print('Simulación completa.')
    # Return result for plotting
    return {
        't': [t1_grid, t2_grid, t3_grid],
        'y': [y1, y2, y3],
        'q': [q1, q2, q3],
        'u': [u1, u2, u3],
        'H': [H1_arr, H2_arr, H3_arr],
        'gamma': [gamma1, gamma2, gamma3],
        'active': [active1, active2, active3],
        'switch_times': [t1, t2],
        'modes': [1, 2, 3],
    } 

def switching_cost(t_switch, y0, qf, t_final):
    print(f"Evaluando switching times: {t_switch}")
    # Penaliza si los tiempos no son crecientes o están fuera de [0, t_final]
    if not (0 < t_switch[0] < t_switch[1] < t_final):
        return 1e10
    result = solve_hybrid_ocp_3mode(y0, qf, t_switch, t_final)
    # Costo: energía acumulada al final
    z_final = result['y'][-1][-1, 1]  # y[-1] es el último modo, [:,1] es z
    return z_final

def optimize_switching_times(y0, qf, t_final, t_switch0=[2.0, 4.0]):
    bounds = [(0.1, t_final-0.2), (0.2, t_final-0.1)]
    res = minimize(
        switching_cost,
        t_switch0,
        args=(y0, qf, t_final),
        bounds=bounds,
        method='L-BFGS-B'
    )
    t_switch_opt = res.x
    result = solve_hybrid_ocp_3mode(y0, qf, t_switch_opt, t_final)
    return t_switch_opt, result

y0 = [0.0, 0.0]
qf = [0.0, 1.0]
t_final = 6.3

print('Iniciando optimización de tiempos de switching...')
t_switch_opt, result = optimize_switching_times(y0, qf, t_final)
print('Tiempos óptimos de switching encontrados:', t_switch_opt)

print('Graficando la solución óptima...')
plot_hybrid_ocp_solution(result, filename='hybrid_ocp_optimal.png')
print('¡Listo! Gráfico guardado como hybrid_ocp_optimal.png') 