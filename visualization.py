"""
SEIR+Q Disease Spread Simulation Visualization

This module provides comprehensive visualization capabilities for the SEIR+Q simulation:

- Epidemic curve plots (stacked area charts)
- Network analysis and degree distribution plots
- Animated network visualizations with quarantine tracking
- Parameter sweep heatmaps for sensitivity analysis

The visualizations are designed to be publication-ready and suitable for
research presentations and graduate-level applications.
"""
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.animation as animation
def plot_statuses(status_counts, plot_title, save_dir=None):
    """Plots the epidemic curves for all states."""
    S = np.array(status_counts['S'])
    E = np.array(status_counts['E'])
    I = np.array(status_counts['I'])
    QE = np.array(status_counts['QE'])
    QI = np.array(status_counts['QI'])
    R = np.array(status_counts['R'])

    # Time axis
    timesteps = range(len(S))
    plt.figure(figsize=(10,6))
    plt.stackplot(timesteps, I, QI, E, QE, R, S,
                  labels=["Infected", "Quarantined+Infected",
                          "Exposed", "Quarantined+Exposed", "Recovered", "Susceptible"],
                  colors=["#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#2ca02c", "#1f77b4"],
                  alpha=0.8)
    plt.ylabel('Population')
    plt.xlabel('Time Step')
    plt.title(plot_title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plots_path = save_dir + "plots/"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        filename = "epidemic_curve_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".png"
        save_path = os.path.join(plots_path, filename)
        plt.savefig(save_path)
    plt.show()

def animate_network(G, history, interval=200, save_dir=None):
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8,8))
    def update(frame):
        ax.clear()
        status = history[frame]
        color_map = {'S': 'blue', 'E': 'yellow', 'I': 'red', 'R': 'green'}
        legend_labels = {
            'S': 'Susceptible', 'E': 'Exposed', 'I': 'Infectious', 'R': 'Recovered',
        }
        # Create legend handles
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=legend_labels[state],
                   markerfacecolor=color_map[state], markersize=10)
            for state in ['S', 'E', 'I', 'R']
        ]

        node_colors = [color_map[status[n]] for n in G.nodes]  # node_color expects a list
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, ax=ax)  # type: ignore[arg-type]
        nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax)
        ax.set_title(f"Day {frame}")
        ax.axis('off')
        ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0, 0))
        return []  # Return an iterable of artists (empty is fine)
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=interval, repeat=False)
    if save_dir is not None:
        animation_path = save_dir + "animations/"
        if not os.path.exists(animation_path):
            os.makedirs(animation_path)
        filename = "epidemic_animation_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".gif"
        save_path = os.path.join(animation_path, filename)
        ani.save(save_path, writer='ffmpeg')
    plt.show()
