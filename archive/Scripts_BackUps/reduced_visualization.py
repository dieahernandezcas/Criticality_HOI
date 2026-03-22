import matplotlib.pyplot as plt
import numpy as np

def plot_reduced_signals(t, states_2d, I_states, E_reduced, P_sim, K_sim, title):
    """Visualización para el modelo reducido."""
    plt.figure(figsize=(15, 10))

    # Variables reducidas (X1, X2)
    plt.subplot(3, 1, 1)
    plt.plot(t, states_2d[:, 0], label='X1')
    plt.plot(t, states_2d[:, 1], label='X2')
    plt.title(f'{title}\nP={P_sim}, K={K_sim}')
    plt.xlabel('Tiempo')
    plt.ylabel('Actividad Reducida')
    plt.legend()
    plt.grid(True)

    # Actividad inhibitoria (I)
    plt.subplot(3, 1, 2)
    for i in range(N_nodes):
        plt.plot(t, I_states[:, i], label=f'I{i+1}')
    plt.title('Actividad Inhibitoria (I)')
    plt.xlabel('Tiempo')
    plt.ylabel('Actividad')
    plt.legend()
    plt.grid(True)

    # Actividad excitatoria reconstruida (E)
    plt.subplot(3, 1, 3)
    for i in range(N_nodes):
        plt.plot(t, E_reduced[i, :], label=f'E{i+1}')
    plt.title('Actividad Excitatoria Reconstruida (E)')
    plt.xlabel('Tiempo')
    plt.ylabel('Actividad')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
