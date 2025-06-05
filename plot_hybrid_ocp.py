import matplotlib.pyplot as plt
import numpy as np

def plot_hybrid_ocp_solution(result, filename='hybrid_ocp_solution.png'):
    """
    Plots the hybrid OCP solution and saves the figure to a file.
    Args:
        result: dict with keys 't', 'y', 'q', 'u', 'H', 'switch_times', 'modes'
        filename: output image filename (default 'hybrid_ocp_solution.png')
    """
    t_list = result['t']
    y_list = result['y']
    q_list = result['q']
    u_list = result['u']
    H_list = result.get('H', None)
    switch_times = result.get('switch_times', [])
    modes = result.get('modes', [1,2,3])

    colors = ['purple', 'blue', 'green']
    linestyles = ['-', '--', '-.']
    mode_labels = [f"q_{{{m}}}" for m in modes]

    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

    # 1. Velocidad (en km/h)
    for i, (t, y) in enumerate(zip(t_list, y_list)):
        if y.shape[1] == 2:  # (v, z)
            v = y[:,0] * 3.6  # m/s to km/h
        else:  # (omega_S, omega_R, z)
            # Puedes agregar aquí la conversión a v si lo deseas
            v = np.zeros_like(t)
        axs[0].plot(t, v, color=colors[i], linestyle=linestyles[i], label=mode_labels[i])
    axs[0].set_ylabel('Velocity (km/h)')
    axs[0].legend()

    # 2. Adjuntos q
    for i, (t, q) in enumerate(zip(t_list, q_list)):
        for j in range(q.shape[1]):
            axs[1].plot(t, q[:,j], color=colors[i], linestyle=linestyles[i], label=f"q{j+1},{modes[i]}")
    axs[1].set_ylabel('q (adjoint)')
    axs[1].legend()

    # 3. Control óptimo u (puede ser escalar o vector)
    for i, (t, u) in enumerate(zip(t_list, u_list)):
        if u.ndim == 1:
            axs[2].plot(t, u, color=colors[i], linestyle=linestyles[i], label=f"u_{modes[i]}")
        else:
            for j in range(u.shape[1]):
                axs[2].plot(t, u[:,j], color=colors[i], linestyle=linestyles[i], label=f"u{j+1},{modes[i]}")
    axs[2].set_ylabel('u (control)')
    axs[2].legend()

    # 4. Torque motor T_M (asumimos es el primer control)
    for i, (t, u) in enumerate(zip(t_list, u_list)):
        if u.ndim == 1:
            axs[3].plot(t, u, color=colors[i], linestyle=linestyles[i], label=f"T_M,{modes[i]}")
        else:
            axs[3].plot(t, u[:,0], color=colors[i], linestyle=linestyles[i], label=f"T_M,{modes[i]}")
    axs[3].set_ylabel('$T_M$ (Nm)')
    axs[3].legend()

    # 5. Hamiltoniano H
    if H_list is not None:
        for i, (t, H) in enumerate(zip(t_list, H_list)):
            axs[4].plot(t, H, color=colors[i], linestyle=linestyles[i], label=f"H_{modes[i]}")
        axs[4].set_ylabel('H')
        axs[4].legend()

    # Líneas verticales y etiquetas en los tiempos de switching
    for idx, ts in enumerate(switch_times):
        for ax in axs:
            ax.axvline(ts, color='red', linestyle=':', alpha=0.7)
        axs[0].text(ts, axs[0].get_ylim()[1]*0.95, f'Switch {idx+1}', color='red', ha='right', va='top', fontsize=9, rotation=90)

    axs[-1].set_xlabel('time (s)')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig) 