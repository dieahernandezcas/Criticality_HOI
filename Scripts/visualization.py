"""
visualization.py
================
Plotting utilities for the Wilson–Cowan higher-order interactions study:

    Hernández et al. (2025) "Higher-order statistical structure emerges from
    nonlinear dynamics without explicit higher-order coupling."

Functions fall into three groups:
    1. Time-series diagnostics  — plot_signals, plot_detailed_signals
    2. Phase-diagram heatmaps   — plot_oscillation_map, plot_metrics_map,
                                   plot_metrics_contour
    3. Line / area plots        — plot_lines, plot_lines_with_area
    4. Avalanche analysis       — detect_avalanches_per_node,
                                   detect_global_avalanches

All interactive figures use Plotly; static figures use Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Basic time-series plots
# ---------------------------------------------------------------------------

def plot_signals(t: np.ndarray, E: np.ndarray, I: np.ndarray,
                 title: str = "Wilson-Cowan Dynamics") -> None:
    """Plot excitatory and inhibitory time series for all nodes.

    Parameters
    ----------
    t : np.ndarray, shape (n_steps,)
        Time vector.
    E : np.ndarray, shape (n_steps, N)
        Excitatory activity matrix (one column per node).
    I : np.ndarray, shape (n_steps, N)
        Inhibitory activity matrix.
    title : str, optional
        Figure title prefix.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, E)
    plt.title(f"{title} — Excitatory Activity (E)")
    plt.xlabel("Time (s)")
    plt.ylabel("Activity")

    plt.subplot(2, 1, 2)
    plt.plot(t, I)
    plt.title(f"{title} — Inhibitory Activity (I)")
    plt.xlabel("Time (s)")
    plt.ylabel("Activity")

    plt.tight_layout()
    plt.show()


def plot_detailed_signals(t: np.ndarray, states: np.ndarray,
                          N_nodes: int, P_sim=None, K_sim=None,
                          K3_sim=None, transient_time: float = 1.0,
                          title: str = "Wilson-Cowan Dynamics") -> None:
    """Multi-panel diagnostic plot for each node.

    For each node, four panels are shown:
        (1) Excitatory time series E(t)
        (2) Inhibitory time series I(t)
        (3) Phase portrait (E vs I)
        (4) Autocorrelation of E (first 200 lags)

    The transient is discarded before plotting.

    Parameters
    ----------
    t : np.ndarray, shape (n_steps,)
        Time vector (seconds).
    states : np.ndarray, shape (n_steps, 2*N_nodes)
        Full state matrix [E_0,...,E_{N-1}, I_0,...,I_{N-1}].
    N_nodes : int
        Number of network nodes.
    P_sim : float or None, optional
        External drive used (for labelling the figure title).
    K_sim : float or None, optional
        Pairwise coupling strength (for labelling).
    K3_sim : float or None, optional
        Higher-order coupling strength (for labelling).
    transient_time : float, optional
        Duration of the transient to discard (default 1.0 s).
    title : str, optional
        Main figure title.
    """
    dt_val = t[1] - t[0]
    transient_steps = int(transient_time / dt_val)

    # Remove transient period
    start_idx = transient_steps
    E = states[start_idx:, :N_nodes]
    I = states[start_idx:, N_nodes:]

    print(f"Total timesteps : {len(states)}")
    print(f"Analysis timesteps: {len(E)}")

    fig, axes = plt.subplots(N_nodes, 4, figsize=(16, 4 * N_nodes))

    colors = ["steelblue", "darkorange", "forestgreen"]
    node_labels = [f"Node {i}" for i in range(N_nodes)]

    for node in range(N_nodes):
        E_node = E[:, node]
        I_node = I[:, node]

        # Print per-node statistics
        print(f"\n{node_labels[node]}:")
        print(f"  E: mean={E_node.mean():.4f}, std={np.std(E_node):.4f}, "
              f"range=[{E_node.min():.4f}, {E_node.max():.4f}]")
        print(f"  I: mean={I_node.mean():.4f}, std={np.std(I_node):.4f}, "
              f"range=[{I_node.min():.4f}, {I_node.max():.4f}]")

        # Panel 1: Excitatory time series
        axes[node, 0].plot(E_node, linewidth=0.8, color=colors[node])
        axes[node, 0].set_ylabel("E", fontsize=11)
        axes[node, 0].set_title(f"{node_labels[node]} — E signal", fontsize=11)
        axes[node, 0].grid(True, alpha=0.3)
        if node == N_nodes - 1:
            axes[node, 0].set_xlabel("Time step", fontsize=10)

        # Panel 2: Inhibitory time series
        axes[node, 1].plot(I_node, linewidth=0.8, color=colors[node], alpha=0.7)
        axes[node, 1].set_ylabel("I", fontsize=11)
        axes[node, 1].set_title(f"{node_labels[node]} — I signal", fontsize=11)
        axes[node, 1].grid(True, alpha=0.3)
        if node == N_nodes - 1:
            axes[node, 1].set_xlabel("Time step", fontsize=10)

        # Panel 3: Phase portrait (E vs I)
        axes[node, 2].plot(E_node, I_node, alpha=0.7, linewidth=0.6,
                           color=colors[node])
        # Mark start (green) and end (red) of the trajectory
        axes[node, 2].scatter(E_node[0], I_node[0], c="green", s=40,
                              zorder=5, edgecolors="black", linewidths=0.5)
        axes[node, 2].scatter(E_node[-1], I_node[-1], c="red", s=40,
                              zorder=5, edgecolors="black", linewidths=0.5)
        axes[node, 2].set_xlabel("E", fontsize=10)
        axes[node, 2].set_ylabel("I", fontsize=10)
        axes[node, 2].set_title(f"{node_labels[node]} — Phase space", fontsize=11)
        axes[node, 2].grid(True, alpha=0.3)

        # Panel 4: Autocorrelation of E (first 200 lags)
        E_centered = E_node - np.mean(E_node)
        if np.std(E_centered) > 1e-6:
            autocorr = np.correlate(E_centered, E_centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / autocorr[0]  # normalize to 1 at lag 0

            axes[node, 3].plot(autocorr[:min(200, len(autocorr))],
                               linewidth=0.8, color=colors[node])
            # Reference line at 0.9 (persistent autocorrelation marker)
            axes[node, 3].axhline(y=0.9, color="r", linestyle="--",
                                  alpha=0.5, linewidth=1)
            axes[node, 3].set_ylabel("Correlation", fontsize=10)
            axes[node, 3].set_title(f"{node_labels[node]} — Autocorrelation",
                                    fontsize=11)
            axes[node, 3].grid(True, alpha=0.3)
            if node == N_nodes - 1:
                axes[node, 3].set_xlabel("Lag", fontsize=10)

    # Figure suptitle: include simulation parameters for traceability
    if P_sim is not None and K_sim is not None:
        plt.suptitle(f"{title}: P={P_sim}, K={K_sim}", fontsize=14,
                     fontweight="bold")
    elif P_sim is not None and K3_sim is not None:
        plt.suptitle(f"{title}: P={P_sim}, K3={K3_sim}", fontsize=14,
                     fontweight="bold")
    else:
        plt.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Phase-diagram heatmap plots
# ---------------------------------------------------------------------------

def plot_oscillation_map(oscillation_map: np.ndarray, P_values: np.ndarray,
                          K_values: np.ndarray,
                          title: str = "Oscillation Detection") -> None:
    """Plot a heatmap of the oscillation detection flag over the (P, K) grid.

    Parameters
    ----------
    oscillation_map : np.ndarray, shape (len(P_values), len(K_values))
        Binary (0/1) matrix; 1 indicates a limit cycle was detected.
    P_values : np.ndarray
        External drive values (y-axis).
    K_values : np.ndarray
        Coupling strength values (x-axis).
    title : str, optional
        Figure title.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(
        oscillation_map,
        origin="lower",
        aspect="auto",
        extent=[K_values[0], K_values[-1], P_values[0], P_values[-1]],
    )
    plt.colorbar(label="Oscillations Detected")
    plt.xlabel(r"$K$")
    plt.ylabel(r"$P$")
    plt.title(title)
    plt.show()


def plot_metrics_map(metric_map: np.ndarray, P_values: np.ndarray,
                     K_values: np.ndarray, title: str = "Metric Map") -> None:
    """Plot a heatmap of a scalar metric over the (P, K) grid.

    Parameters
    ----------
    metric_map : np.ndarray, shape (len(P_values), len(K_values))
        Matrix of metric values (e.g., TC−DTC or κ_{123}).
    P_values : np.ndarray
    K_values : np.ndarray
    title : str, optional
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(
        metric_map,
        origin="lower",
        aspect="auto",
        extent=[K_values[0], K_values[-1], P_values[0], P_values[-1]],
    )
    plt.colorbar(label=title)
    plt.xlabel(r"$K$")
    plt.ylabel(r"$P$")
    plt.title(title)
    plt.show()


def plot_metrics_contour(matrix: np.ndarray, P_values: np.ndarray,
                          K_values: np.ndarray, title: str = "Metric Map",
                          colorscale: str = "Inferno",
                          n_contours: int = 15) -> None:
    """Interactive Plotly contour map of a metric over the (P, K) grid.

    Saves a static PNG to ``imgs/<title>.png`` in addition to showing the
    interactive figure.

    Parameters
    ----------
    matrix : np.ndarray, shape (len(P_values), len(K_values))
        Metric values (rows = P, columns = K).
    P_values : np.ndarray
        External drive values (y-axis).
    K_values : np.ndarray
        Coupling strength values (x-axis).
    title : str, optional
        Figure title and output filename stem.
    colorscale : str, optional
        Plotly colorscale name (default "Inferno").
    n_contours : int, optional
        Number of contour levels (default 15).
    """
    z_min = float(np.nanmin(matrix))
    z_max = float(np.nanmax(matrix))

    fig = go.Figure(data=go.Contour(
        z=matrix,
        x=K_values,
        y=P_values,
        colorscale=colorscale,
        contours=dict(
            start=z_min,
            end=z_max,
            size=(z_max - z_min) / n_contours,
            coloring="heatmap",   # filled contour with overlaid isolines
            showlines=True,
        ),
        colorbar=dict(title=title),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title="P",
        width=850,
        height=700,
    )

    fig.show()
    fig.write_image(f"imgs/{title}.png")


# ---------------------------------------------------------------------------
# Line and area plots
# ---------------------------------------------------------------------------

def plot_lines(matrix: np.ndarray, x_values: np.ndarray = None,
               title: str = "Metric Curves") -> None:
    """Plot each row of *matrix* as a separate line using Plotly.

    Useful for comparing how a metric evolves as a function of K for
    different fixed values of P (or a_E).

    Parameters
    ----------
    matrix : np.ndarray, shape (n_lines, n_points)
        Each row is one curve to plot.
    x_values : np.ndarray or None, optional
        x-axis values (default: integer indices).
    title : str, optional
        Figure title.
    """
    matrix = np.array(matrix)
    n_lines, n_points = matrix.shape

    if x_values is None:
        x_values = np.arange(n_points)

    fig = go.Figure()

    for i in range(n_lines):
        y = matrix[i]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y,
            mode="lines+markers",
            name=f"Line {i + 1}",
            marker=dict(size=3, color=y),
            line=dict(width=4),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title="Metric value",
        width=950,
        height=600,
    )
    fig.show()


def plot_lines_with_area(matrix: np.ndarray, x_values: np.ndarray = None,
                          title: str = "Metric Curves") -> None:
    """Plot each row of *matrix* as a filled area trace using Plotly.

    Similar to ``plot_lines`` but with semi-transparent fills to the zero
    baseline, making it easier to compare magnitudes across curves.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_lines, n_points)
    x_values : np.ndarray or None, optional
    title : str, optional
    """
    matrix = np.array(matrix)
    n_lines, n_points = matrix.shape

    if x_values is None:
        x_values = np.arange(n_points)

    # Color palette: orange, blue, purple, green (cycles for >4 lines)
    line_colors = [
        "rgb(255,120,0)",
        "rgb(0,180,255)",
        "rgb(180,0,255)",
        "rgb(0,200,120)",
    ]

    fig = go.Figure()

    for i in range(n_lines):
        y = matrix[i]
        color = line_colors[i % len(line_colors)]
        # Convert to rgba for the semi-transparent fill
        fill_color = color.replace("rgb", "rgba").replace(")", ",0.25)")

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y,
            mode="lines",
            name=f"Line {i + 1}",
            line=dict(color=color, width=4),
            fill="tozeroy",
            fillcolor=fill_color,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title="Metric value",
        width=950,
        height=600,
    )
    fig.show()


# ---------------------------------------------------------------------------
# Avalanche detection and visualization
# ---------------------------------------------------------------------------

def detect_avalanches_per_node(t: np.ndarray, states: np.ndarray,
                                N_nodes: int, transient_time: float = 1.0,
                                bin_size: float = 0.02) -> None:
    """Detect and plot neuronal avalanche statistics for each node.

    Avalanche detection procedure:
        1. Remove transient.
        2. Bin the excitatory signal with window *bin_size*.
        3. Threshold at the mean of the binned signal.
        4. Identify contiguous above-threshold segments (avalanches).
        5. Record size (sum of binned activity) and duration (number of bins).

    Parameters
    ----------
    t : np.ndarray, shape (n_steps,)
        Time vector.
    states : np.ndarray, shape (n_steps, 2*N_nodes)
        Full state matrix.
    N_nodes : int
        Number of network nodes.
    transient_time : float, optional
        Duration to discard as transient (default 1.0 s).
    bin_size : float, optional
        Temporal bin width in seconds (default 0.02 s).
    """
    dt_val = t[1] - t[0]
    transient_steps = int(transient_time / dt_val)

    E = states[transient_steps:, :N_nodes]

    bin_steps = int(bin_size / dt_val)
    print(f"\nBin size: {bin_size} s ({bin_steps} steps)")

    fig, axes = plt.subplots(N_nodes, 2, figsize=(10, 4 * N_nodes))

    for node in range(N_nodes):
        signal = E[:, node]

        # --- Bin the signal: compute mean activity in each window ---
        n_bins = len(signal) // bin_steps
        signal_trim = signal[:n_bins * bin_steps]
        signal_binned = signal_trim.reshape(n_bins, bin_steps).mean(axis=1)

        # --- Threshold at the mean (common in avalanche analysis) ---
        threshold = np.mean(signal_binned)
        above = signal_binned > threshold

        # --- Detect avalanches (contiguous above-threshold runs) ---
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

        # --- Histograms of avalanche size and duration ---
        axes[node, 0].hist(sizes, bins=30, density=True)
        axes[node, 0].set_title(f"Node {node} — Avalanche Size Distribution")
        axes[node, 0].set_xlabel("S")
        axes[node, 0].set_ylabel("P(S)")

        axes[node, 1].hist(durations, bins=30, density=True)
        axes[node, 1].set_title(f"Node {node} — Avalanche Duration Distribution")
        axes[node, 1].set_xlabel("T (bins)")
        axes[node, 1].set_ylabel("P(T)")

    plt.tight_layout()
    plt.show()


def detect_global_avalanches(t: np.ndarray, states: np.ndarray,
                              N_nodes: int, transient_time: float = 2.0,
                              bin_size: float = 0.02) -> None:
    """Detect and plot avalanche statistics for the summed (global) signal.

    Same procedure as ``detect_avalanches_per_node`` but applied to
    E_global = sum_i E_i, providing network-level avalanche statistics.
    Results are shown as complementary CDFs (log–log scale).

    Parameters
    ----------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2*N_nodes)
    N_nodes : int
    transient_time : float, optional
        Duration to discard (default 2.0 s).
    bin_size : float, optional
        Temporal bin width in seconds (default 0.02 s).
    """
    dt_val = t[1] - t[0]
    transient_steps = int(transient_time / dt_val)

    E = states[transient_steps:, :N_nodes]

    # Sum excitatory activity across all nodes → global signal
    global_signal = np.sum(E, axis=1)

    bin_steps = int(bin_size / dt_val)
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

    print("\nGLOBAL SIGNAL AVALANCHES")
    print(f"  Avalanches detected: {len(sizes)}")

    plt.figure(figsize=(10, 4))

    def plot_ccdf(data: np.ndarray, label: str) -> None:
        """Plot the complementary CDF (CCDF) on a log–log scale."""
        data = data[data > 0]
        values, counts = np.unique(data, return_counts=True)
        prob = counts / np.sum(counts)
        ccdf = 1.0 - np.cumsum(prob)
        mask = ccdf > 0
        plt.loglog(values[mask], ccdf[mask], "o", label=label, linewidth=2.5)

    plot_ccdf(sizes, "Size")
    plot_ccdf(durations, "Duration")

    plt.legend()
    plt.title("Global Avalanche Statistics — CCDF")
    plt.xlabel("Value")
    plt.ylabel("P(X > x)")
    plt.tight_layout()
    plt.show()
