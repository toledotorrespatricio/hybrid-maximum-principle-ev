from hybrid_maximum_principle import solve_hybrid_ocp_3mode
from plot_hybrid_ocp import plot_hybrid_ocp_solution

y0 = [0.0, 0.0]
qf = [0.0, 1.0]
t_final = 6.3
t_switch = [1.4, 1.7]  # Valores manuales sugeridos

print(f'Corriendo (debug rápido) con tiempos de switching manuales: {t_switch}')
result = solve_hybrid_ocp_3mode(y0, qf, t_switch, t_final, num_points=3, rtol=1e-2, atol=1e-5)
plot_hybrid_ocp_solution(result, filename='hybrid_ocp_manual_switch.png')
print('¡Listo! Gráfico guardado como hybrid_ocp_manual_switch.png') 