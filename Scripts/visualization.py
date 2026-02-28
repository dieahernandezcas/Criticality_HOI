# functions/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_signals(t, E, I, title="Wilson-Cowan Dynamics"):
    """Plot excitatory and inhibitory signals."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, E)
    plt.title(f"{title} - Excitatory Signals (E)")
    plt.xlabel("Time")
    plt.ylabel("Activity")

    plt.subplot(2, 1, 2)
    plt.plot(t, I)
    plt.title(f"{title} - Inhibitory Signals (I)")
    plt.xlabel("Time")
    plt.ylabel("Activity")

    plt.tight_layout()
    plt.show()

def plot_oscillation_map(oscillation_map, P_values, K_values, title="Oscillation Detection"):
    """Plot a heatmap of oscillation detection over P and K values."""
    plt.figure(figsize=(10, 8))
    plt.imshow(oscillation_map, origin='lower', aspect='auto',
               extent=[K_values[0], K_values[-1], P_values[0], P_values[-1]])
    plt.colorbar(label='Oscillations Detected')
    plt.xlabel(r'$K$')
    plt.ylabel(r'$P$')
    plt.title(title)
    plt.show()

def plot_metrics_map(metric_map, P_values, K_values, title="Metric Map"):
    """Plot a heatmap of a metric over P and K values."""
    plt.figure(figsize=(10, 8))
    plt.imshow(metric_map, origin='lower', aspect='auto',
               extent=[K_values[0], K_values[-1], P_values[0], P_values[-1]])
    plt.colorbar(label=title)
    plt.xlabel(r'$K$')
    plt.ylabel(r'$P$')
    plt.title(title)
    plt.show()

def plot_detailed_signals(t, states, N_nodes, P_sim=None, K_sim=None, K3_sim=None, transient_time=1.0, title="Wilson-Cowan Dynamics"):
    """
    Plot detailed signals for each node in the Wilson-Cowan network, including E and I signals,
    phase space, and autocorrelation.

    Parameters:
    -----------
    t : ndarray, shape (n_steps,)
        Time vector.
    states : ndarray, shape (n_steps, 2*N)
        Time series of state vectors.
    N_nodes : int
        Number of nodes in the network.
    P_sim : float, optional
        External input strength.
    K_sim : float, optional
        Coupling strength for pairwise interactions.
    K3_sim : float, optional
        Higher-order coupling strength.
    transient_time : float, optional
        Time to discard as transient (default: 1.0 seconds).
    title : str, optional
        Title for the plot (default: "Wilson-Cowan Dynamics").
    """
    dt = t[1] - t[0]  # Calculate time step from time vector
    transient_steps = int(transient_time / dt)

    # Remove transient
    start_idx = transient_steps
    E = states[start_idx:, :N_nodes]
    I = states[start_idx:, N_nodes:]

    print(f"Total timesteps: {len(states)}")
    print(f"Analysis timesteps: {len(E)}")

    # Visualization for all nodes
    fig, axes = plt.subplots(N_nodes, 4, figsize=(16, 4 * N_nodes))

    colors = ['steelblue', 'darkorange', 'forestgreen']
    node_labels = [f'Node {i}' for i in range(N_nodes)]

    for node in range(N_nodes):
        E_node = E[:, node]
        I_node = I[:, node]

        # Statistics
        print(f"\n{node_labels[node]}:")
        print(f"  E: mean={E_node.mean():.4f}, std={np.std(E_node):.4f}, range=[{E_node.min():.4f}, {E_node.max():.4f}]")
        print(f"  I: mean={I_node.mean():.4f}, std={np.std(I_node):.4f}, range=[{I_node.min():.4f}, {I_node.max():.4f}]")

        # Excitatory time series
        axes[node, 0].plot(E_node, linewidth=0.8, color=colors[node])
        axes[node, 0].set_ylabel("E", fontsize=11)
        axes[node, 0].set_title(f"{node_labels[node]} - E signal", fontsize=11)
        axes[node, 0].grid(True, alpha=0.3)
        if node == N_nodes - 1:
            axes[node, 0].set_xlabel("Time step", fontsize=10)

        # Inhibitory time series
        axes[node, 1].plot(I_node, linewidth=0.8, color=colors[node], alpha=0.7)
        axes[node, 1].set_ylabel("I", fontsize=11)
        axes[node, 1].set_title(f"{node_labels[node]} - I signal", fontsize=11)
        axes[node, 1].grid(True, alpha=0.3)
        if node == N_nodes - 1:
            axes[node, 1].set_xlabel("Time step", fontsize=10)

        # Phase space (E vs I)
        axes[node, 2].plot(E_node, I_node, alpha=0.7, linewidth=0.6, color=colors[node])
        axes[node, 2].scatter(E_node[0], I_node[0], c='green', s=40, zorder=5, edgecolors='black', linewidths=0.5)
        axes[node, 2].scatter(E_node[-1], I_node[-1], c='red', s=40, zorder=5, edgecolors='black', linewidths=0.5)
        axes[node, 2].set_xlabel("E", fontsize=10)
        axes[node, 2].set_ylabel("I", fontsize=10)
        axes[node, 2].set_title(f"{node_labels[node]} - Phase space", fontsize=11)
        axes[node, 2].grid(True, alpha=0.3)

        # Autocorrelation
        E_centered = E_node - np.mean(E_node)
        if np.std(E_centered) > 1e-6:
            autocorr = np.correlate(E_centered, E_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            axes[node, 3].plot(autocorr[:min(200, len(autocorr))], linewidth=0.8, color=colors[node])
            axes[node, 3].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, linewidth=1)
            axes[node, 3].set_ylabel("Correlation", fontsize=10)
            axes[node, 3].set_title(f"{node_labels[node]} - Autocorrelation", fontsize=11)
            axes[node, 3].grid(True, alpha=0.3)
            if node == N_nodes - 1:
                axes[node, 3].set_xlabel("Lag", fontsize=10)

    # Set the main title based on the parameters provided
    if P_sim is not None and K_sim is not None:
        plt.suptitle(f"{title}: P={P_sim}, K={K_sim}", fontsize=14, fontweight='bold')
    elif P_sim is not None and K3_sim is not None:
        plt.suptitle(f"{title}: P={P_sim}, K3={K3_sim}", fontsize=14, fontweight='bold')
    else:
        plt.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()




def detect_avalanches_per_node(t, states, N_nodes, transient_time=1.0, bin_size=0.02):
    """
    Detect avalanches per node using the methodology of the Wilson-Cowan critical paper.
    
    Parameters
    ----------
    t : array
        Time vector
    states : array (n_steps, 2*N)
        State matrix
    N_nodes : int
        Number of nodes
    transient_time : float
        Time to discard
    bin_size : float
        Temporal bin size (in seconds)
    """
    
    dt = t[1] - t[0]
    transient_steps = int(transient_time / dt)
    
    # Remove transient
    E = states[transient_steps:, :N_nodes]
    t = t[transient_steps:]
    
    bin_steps = int(bin_size / dt)
    
    print(f"\nBin size: {bin_size} s ({bin_steps} steps)")
    
    fig, axes = plt.subplots(N_nodes, 2, figsize=(10, 4*N_nodes))
    
    for node in range(N_nodes):
        
        signal = E[:, node]
        
        # --- Binning ---
        n_bins = len(signal) // bin_steps
        signal_trim = signal[:n_bins * bin_steps]
        signal_binned = signal_trim.reshape(n_bins, bin_steps).mean(axis=1)
        
        # --- Threshold (mean) ---
        threshold = np.mean(signal_binned)
        
        # --- Avalanche detection ---
        above = signal_binned > threshold
        
        sizes = []
        durations = []
        
        current_size = 0
        current_duration = 0
        
        for val, is_active in zip(signal_binned, above):
            if is_active:
                current_size += val
                current_duration += 1
            else:
                if current_duration > 0:
                    sizes.append(current_size)
                    durations.append(current_duration)
                current_size = 0
                current_duration = 0
        
        sizes = np.array(sizes)
        durations = np.array(durations)
        
        print(f"\nNode {node}")
        print(f"  Avalanches detected: {len(sizes)}")


        axes[node, 0].hist(sizes, bins=30, density=True)
        axes[node, 0].set_title(f"Node {node} - Size distribution")
        axes[node, 0].set_xlabel("S")
        axes[node, 0].set_ylabel("P(S)")
        
        axes[node, 1].hist(durations, bins=30, density=True)
        axes[node, 1].set_title(f"Node {node} - Duration distribution")
        axes[node, 1].set_xlabel("T")
        axes[node, 1].set_ylabel("P(T)")

    
    plt.tight_layout()
    plt.show()



def detect_global_avalanches(t, states, N_nodes,
                             transient_time=2.0,
                             bin_size=0.02):

    dt = t[1] - t[0]
    transient_steps = int(transient_time / dt)

    E = states[transient_steps:, :N_nodes]

    global_signal = np.sum(E, axis=1)

    bin_steps = int(bin_size / dt)
    n_bins = len(global_signal) // bin_steps

    signal_trim = global_signal[:n_bins * bin_steps]
    signal_binned = signal_trim.reshape(n_bins, bin_steps).mean(axis=1)

    threshold = np.mean(signal_binned)

    above = signal_binned > threshold

    sizes = []
    durations = []

    current_size = 0
    current_duration = 0

    for val, active in zip(signal_binned, above):

        if active:
            current_size += val
            current_duration += 1
        else:
            if current_duration > 0:
                sizes.append(current_size)
                durations.append(current_duration)
            current_size = 0
            current_duration = 0

    sizes = np.array(sizes)
    durations = np.array(durations)

    print("\nGLOBAL SIGNAL")
    print("Avalanches:", len(sizes))

    plt.figure(figsize=(10,4))

    def plot_ccdf(data, label):
        data = data[data > 0]
        values, counts = np.unique(data, return_counts=True)
        prob = counts / np.sum(counts)
        ccdf = 1.0 - np.cumsum(prob)
        mask = ccdf > 0
        plt.loglog(values[mask], ccdf[mask], 'o', label=label, linewidth = 2.5)

    plot_ccdf(sizes, "Size")
    plot_ccdf(durations, "Duration")

    plt.legend()
    plt.title("Global Avalanche Statistics")
    plt.xlabel("Value")
    plt.ylabel("CCDF")
    plt.show()



def plot_metrics_contour(matrix, P_values, K_values, title="Metric Map",
                         colorscale="Inferno", n_contours=15):
    """
    Plot 2D interactive contour map using Plotly.
    
    Parameters
    ----------
    matrix : 2D np.array
        Matriz (filas=P, columnas=K)
    P_values : array
        Valores eje Y
    K_values : array
        Valores eje X
    title : str
    colorscale : str or custom list
    n_contours : int
        Número de curvas de nivel
    """

    z_min = np.nanmin(matrix)
    z_max = np.nanmax(matrix)

    fig = go.Figure(data=go.Contour(
        z=matrix,
        x=K_values,
        y=P_values,
        colorscale=colorscale,
        contours=dict(
            start=z_min,
            end=z_max,
            size=(z_max - z_min) / n_contours,
            coloring='heatmap',  # mantiene fondo coloreado
            showlines=True
        ),
        colorbar=dict(title=title)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='K',
        yaxis_title='P',
        width=850,
        height=700
    )

    fig.show()



def plot_lines(matrix, x_values=None, title="Metric Curves"):
    
    matrix = np.array(matrix)
    n_lines, n_points = matrix.shape
    
    if x_values is None:
        x_values = np.arange(n_points)

    # Colormap estilo wildfire
    colorscale_wildfire = [
        [0.0, 'black'],
        [0.3, 'darkred'],
        [0.6, 'orange'],
        [0.85, 'yellow'],
        [1.0, 'white']
    ]

    fig = go.Figure()

    for i in range(n_lines):
        y = matrix[i]

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y,
            mode='lines+markers',
            name=f"Line {i+1}",
            marker=dict(
                size=3,
                color=y,
                #colorscale=colorscale_wildfire,
                #showscale=(i == 0),  # solo una colorbar
                #colorbar=dict(title="Value") if i == 0 else None
            ),
            line=dict(width=4)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title="Metric value",
        width=950,
        height=600
    )

    fig.show()


def plot_lines_with_area(matrix, x_values=None, title="Metric Curves"):

    matrix = np.array(matrix)
    n_lines, n_points = matrix.shape

    if x_values is None:
        x_values = np.arange(n_points)

    # Colores base para cada línea
    line_colors = [
        'rgb(255,120,0)',   # naranja
        'rgb(0,180,255)',   # azul
        'rgb(180,0,255)',   # púrpura
        'rgb(0,200,120)'    # verde
    ]

    fig = go.Figure()

    for i in range(n_lines):
        y = matrix[i]
        color = line_colors[i % len(line_colors)]

        # convertir a rgba para el área con opacidad
        fill_color = color.replace('rgb', 'rgba').replace(')', ',0.25)')

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y,
            mode='lines',
            name=f"Line {i+1}",
            line=dict(color=color, width=4),
            fill='tozeroy',
            fillcolor=fill_color,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title="Metric value",
        width=950,
        height=600
    )

    fig.show()