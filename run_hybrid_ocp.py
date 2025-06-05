from hybrid_maximum_principle import optimize_switching_times
from plot_hybrid_ocp import plot_hybrid_ocp_solution

y0 = [0.0, 0.0]
qf = [0.0, 1.0]
t_final = 6.3

print('Iniciando optimización de tiempos de switching...')
t_switch_opt, result = optimize_switching_times(y0, qf, t_final)
print('Tiempos óptimos de switching encontrados:', t_switch_opt)

print('Graficando la solución óptima...')
plot_hybrid_ocp_solution(result, filename='hybrid_ocp_optimal.png')
print('¡Listo! Gráfico guardado como hybrid_ocp_optimal.png') 