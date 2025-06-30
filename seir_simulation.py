import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from enum import Enum
import os
import argparse
import datetime

class GraphOptions:
    def __init__(self, graph_type):
        self.graph_type = graph_type

class GraphType(Enum):
    BINOMIAL = 1 
    SMALL_WORLD = 2
    PREFERENTIAL_ATTACHMENT = 3
    STOCHASTIC_BLOCK = 4
    GEOMETRIC = 5
    POWERLAW_CLUSTER = 6

def build_graph(graph_options, num_nodes, avg_num_edges, initial_infected):
    """Builds a social network graph of the specified type and initializes SEIR states."""
    graph_type = graph_options.graph_type
    G = None
    if graph_type == GraphType.BINOMIAL:
        G = nx.erdos_renyi_graph(n=num_nodes, p=10/num_nodes)
    elif graph_type == GraphType.SMALL_WORLD:
        G = nx.watts_strogatz_graph(n=num_nodes, k=avg_num_edges, p=0.1)
    elif graph_type == GraphType.PREFERENTIAL_ATTACHMENT:
        G = nx.barabasi_albert_graph(n=num_nodes, k=avg_num_edges)
    elif graph_type == GraphType.STOCHASTIC_BLOCK:
        # TODO, default to small world
        G = nx.watts_strogatz_graph(n=num_nodes, k=avg_num_edges, p=0.1)
    elif graph_type == GraphType.GEOMETRIC:
        G = nx.random_geometric_graph(n=num_nodes, radius=0.1)
    elif graph_type == GraphType.POWERLAW_CLUSTER:
        G = nx.powerlaw_cluster_graph(n=num_nodes, m=avg_num_edges, p=0.3)
    if G is None:
        raise TypeError(f"Graph type '{graph_type}' is not supported")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # Initialize states in Graph.
    for node in G.nodes():
        G.nodes[node]["status"] = "S"
        G.nodes[node]["quarantined"] = False

    # Add initial_infected random infected nodes
    for infected_node in random.sample(list(G.nodes()), initial_infected):
        G.nodes[infected_node]["status"] = "I" 
        G.nodes[infected_node]["infected_days"] = 0

    return G

def get_status_counts(G):
    """Returns counts of each SEIR+Q state in the graph."""
    s_count, e_count, i_count, qe_count, qi_count, r_count = 0,0,0,0,0,0
    for node in G.nodes:
        status = G.nodes[node]["status"]
        quarantined = G.nodes[node].get("quarantined", False)

        if status == "S":
            s_count += 1
        if status == "E" and not quarantined:
            e_count += 1
        if status == "E" and quarantined:
            qe_count += 1
        if status == "I" and not quarantined:
            i_count += 1
        if status == "I" and quarantined:
            qi_count += 1
        if status == "R":
            r_count += 1
    
    return s_count, e_count, i_count, qe_count, qi_count, r_count

def simulate_step(G, status_counts, p_transmission=0.05, p_quarantine_exposed=0.1, p_quarantine_infected=0.5, exposure_days=3, recovery_days=10):
    """Simulates one time step of SEIR+Q dynamics on the graph."""
    newly_exposed = []
    newly_infected = []
    for node in G.nodes:
        node_status = G.nodes[node]["status"]
        if node_status == "I":
            G.nodes[node]["infected_days"] += 1
            if random.random() < p_quarantine_infected:
                G.nodes[node]["quarantined"] = True
            if G.nodes[node]["infected_days"] > recovery_days:
                G.nodes[node]["status"] = "R"
                continue
            if G.nodes[node]["quarantined"]: # Doesn't infect other nodes
                continue
            for neighbor in G.neighbors(node):
                neighbor_status = G.nodes[neighbor]["status"]
                if neighbor_status == "S" and neighbor not in newly_exposed and random.random() < p_transmission:
                    newly_exposed.append(neighbor)
        elif node_status == "E":
            G.nodes[node]["exposed_days"] += 1
            if random.random() < p_quarantine_exposed:
                G.nodes[node]["quarantined"] = True
            if G.nodes[node]["exposed_days"] > exposure_days:
                newly_infected.append(node)

    for exposed_node in newly_exposed:
        G.nodes[exposed_node]["status"] = "E"
        G.nodes[exposed_node]["exposed_days"] = 0

    for infected_node in newly_infected:
        G.nodes[infected_node]["status"] = "I"
        G.nodes[infected_node]["exposed_days"] = 0
        G.nodes[infected_node]["infected_days"] = 0
    
    S,E,I,QE,QI,R = get_status_counts(G)
    status_counts['S'].append(S)
    status_counts['E'].append(E)
    status_counts['I'].append(I)
    status_counts['QE'].append(QE)
    status_counts['QI'].append(QI)
    status_counts['R'].append(R)
    return status_counts

def plot_statuses(status_counts, param_str, save_dir=None):
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
                  labels=["Infected", "Quarantined+Infected", "Exposed", "Quarantined+Exposed", "Recovered", "Susceptible"],
                  colors=["#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#2ca02c", "#1f77b4"],
                  alpha=0.8)
    plt.ylabel('Population')
    plt.xlabel('Time Step')
    plt.title(param_str)
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

def run_simulation(epochs,
                   graph_options,
                   p_transmission, 
                   p_quarantine_exposed, 
                   p_quarantine_infected, 
                   exposure_days, 
                   recovery_days, 
                   num_nodes,
                   avg_num_edges,
                   initial_infected=1,
                   save_dir=None,
                   plot_population=False,
                   record_history=False):
    """Runs the SEIR+Q simulation and returns the status counts."""
    G = build_graph(graph_options, num_nodes, avg_num_edges, initial_infected)

    # Counts of each status at each epoch, for plotting.
    status_counts = {k: [] for k in ["S", "E", "I", "QE", "QI", "R"]}
    history = [] if record_history else None

    for _ in range(epochs):
        if history is not None:
            history.append({n: G.nodes[n]["status"] for n in G.nodes})
        status_counts = simulate_step(G, 
                                      status_counts, 
                                      p_transmission,
                                      p_quarantine_exposed,
                                      p_quarantine_infected,
                                      exposure_days,
                                      recovery_days)
    param_str = (f"p_trans={p_transmission:.2f}, p_qE={p_quarantine_exposed:.2f}, p_qI={p_quarantine_infected:.2f},\n"
                    f"exp_days={exposure_days}, rec_days={recovery_days}, nodes={num_nodes}, edges={avg_num_edges}")
    if plot_population:
        plot_statuses(status_counts, param_str, save_dir)
    return status_counts, history, G

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

class NumRange:
    """Bounded Int type for argparse typechecking and better-than-default error messaging."""
    def __init__(self, low=None, high=None, var_type=type[int]):
        self.low = low
        self.high = high
        self.expected_type = var_type
    def __call__(self, value):
        try:
            if self.expected_type is int:
                value = int(value)
            elif self.expected_type is float:
                value = float(value)
        except ValueError:
            raise self.type_exception(value)
        if (self.low is not None and value < self.low) or (self.high is not None and value > self.high):
            raise self.range_exception(value)
        return value
    def type_exception(self, val_type):
        return argparse.ArgumentTypeError(val_type, f'Received {val_type} of type {type(val_type)}; expected {self.expected_type}')
    def range_exception(self, value):
        if self.low is not None and self.high is not None:
            return argparse.ArgumentError(value, f"Must be a {self.expected_type} in the range [{self.low}, {self.high}]")
        elif self.low is not None:
            return argparse.ArgumentError(value, f"Must be an {self.expected_type} >= {self.low}")
        elif self.high is not None:
            return argparse.ArgumentError(value, f"Must be an {self.expected_type} <= {self.high}")
        else:
            return argparse.ArgumentError(value, "Error with argument in NumRange")

def main():
    parser = argparse.ArgumentParser(description="SEIR+Q Disease Spread Simulation on Social Networks")
    parser.add_argument('--graph_type', type=str, default='SMALL_WORLD', choices=[g.name for g in GraphType], help='Type of network graph')
    parser.add_argument('--num_nodes', type=NumRange(2**3, 2**12), default=256, help='Number of nodes in the network')
    parser.add_argument('--avg_num_edges', type=NumRange(2**2, 2**6), default=8, help='Average degree/inter-cluster edges')
    parser.add_argument('--initial_infected', type=NumRange(high=2**4), default=4, help='Initial number of infected individuals')
    parser.add_argument('--epochs', type=NumRange(20, 251), default=80, help='Number of simulation steps')
    parser.add_argument('--p_transmission', type=NumRange(0.0, 1.0, var_type=float), default=0.2, help='Transmission probability per contact')
    parser.add_argument('--p_qE', type=NumRange(0.0, 1.0, var_type=float), default=0.05, help='Quarantine probability for exposed')
    parser.add_argument('--p_qI', type=NumRange(0.0, 1.0, var_type=float), default=0.4, help='Quarantine probability for infected')
    parser.add_argument('--exposure_days', type=NumRange(high=10), default=3, help='Days in exposed state before infectious')
    parser.add_argument('--recovery_days', type=NumRange(2,30), default=14, help='Days in infectious state before recovery')
    parser.add_argument('--plot_population', action='store_true', help='Plot population against time')
    parser.add_argument('--plot_animation', action='store_true', help='Create an animated network visualization')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots')
    args = parser.parse_args()
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    graph_options = GraphOptions(GraphType[args.graph_type])
    status_counts, history, G = run_simulation(
        epochs=args.epochs,
        graph_options=graph_options,
        p_transmission=args.p_transmission,
        p_quarantine_exposed=args.p_qE,
        p_quarantine_infected=args.p_qI,
        exposure_days=args.exposure_days,
        recovery_days=args.recovery_days,
        num_nodes=args.num_nodes,
        avg_num_edges=args.avg_num_edges,
        initial_infected=args.initial_infected,
        plot_population=args.plot_population,
        record_history=args.plot_animation,
        save_dir=args.save_dir
    )
    if history is not None and args.plot_animation:
        animate_network(G, history, interval=200, save_dir=args.save_dir)

if __name__ == "__main__":
    main()