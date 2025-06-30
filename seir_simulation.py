import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display
import time
import numpy as np
from enum import Enum
import itertools
import functools
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
    # G = nx.Graph()
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
    newly_quarantined = []
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

def plot_statuses(status_counts, p_transmission, p_quarantine_exposed, p_quarantine_infected, exposure_days, recovery_days, num_nodes, avg_num_edges, save_dir=None, fractional=False):
    """Plots the epidemic curves for all states."""

    S = np.array(status_counts['S'])
    E = np.array(status_counts['E'])
    I = np.array(status_counts['I'])
    QE = np.array(status_counts['QE'])
    QI = np.array(status_counts['QI'])
    R = np.array(status_counts['R'])

    # Normalize to fractions
    S_frac = S / num_nodes
    E_frac = E / num_nodes
    I_frac = I / num_nodes
    QE_frac = QE / num_nodes
    QI_frac = QI / num_nodes
    R_frac = R / num_nodes
    
    # Time axis
    timesteps = range(len(S))
    plt.figure(figsize=(10,6))
    if fractional:
        plt.stackplot(timesteps, I_frac, QI_frac, E_frac, QE_frac, R_frac, S_frac, 
                      labels=["Infected", "Quarantined+Infected", "Exposed", "Quarantined+Exposed", "Recovered", "Susceptible"],
                      colors=["#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#2ca02c", "#1f77b4"],
                      alpha=0.8)
        plt.ylabel('Fraction of population')
    else:
        plt.stackplot(timesteps, I, QI, E, QE, R, S, 
                      labels=["Infected", "Quarantined+Infected", "Exposed", "Quarantined+Exposed", "Recovered", "Susceptible"],
                      colors=["#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#2ca02c", "#1f77b4"],
                      alpha=0.8)
        plt.ylabel('Population')
    plt.xlabel('Time Step')
    plt.title(f'p_transmission={p_transmission:.2f}, p_quarantine_exposed={p_quarantine_exposed:.1f}, p_quarantine_infected={p_quarantine_infected:.1f},\n exposure_days={exposure_days}, recovery_days={recovery_days}, num_nodes={num_nodes}, avg_num_edges={avg_num_edges}')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        file_name = "epidemic_curve_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".png"
        save_path = os.path.join(save_dir, file_name) if save_dir else None
        plt.savefig(save_path)
    plt.show()
git
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
                   prebuilt_graph=None,
                   save_dir=None,
                   do_plot=False):
    """Runs the SEIR+Q simulation and returns the status counts."""
    G = build_graph(graph_options, num_nodes, avg_num_edges, initial_infected)

    # Counts of each status at each epoch, for plotting.
    status_counts = {
        'S': [],
        'E': [],
        'I': [],
        'QE': [],
        'QI': [],
        'R': []
    }

    for epoch in range(epochs):
        status_counts = simulate_step(G, 
                                      status_counts, 
                                      p_transmission=p_transmission, 
                                      p_quarantine_exposed=p_quarantine_exposed,
                                      p_quarantine_infected=p_quarantine_infected,
                                      exposure_days=exposure_days,
                                      recovery_days=recovery_days)
    if do_plot:
        display.clear_output(wait=True)
        plot_statuses(status_counts,
                      p_transmission,
                      p_quarantine_exposed,
                      p_quarantine_infected,
                      exposure_days,
                      recovery_days,
                      num_nodes,
                      avg_num_edges,
                      save_dir)

def num_param_combinations(param_choices):
    return functools.reduce(lambda x,y: x * len(y), param_choices.values(), 1)

def plot_variable_params(param_choices, graph_type, epochs=80, sleep_time=0.5):
    param_values = [param_choices[key] for key in param_choices]
    for (p_transmission,
     p_quarantine_exposed,
     p_quarantine_infected,
     exposure_days,
     recovery_days,
     num_clusters,
     nodes_per_cluster,
     inter_cluster_edges,
     initial_infected) in itertools.product(*param_values):
        # Too slow to compute
        if num_clusters * nodes_per_cluster > 5000:
            continue
        # Doesn't match a realistic quarantine scenario
        if p_quarantine_exposed > p_quarantine_infected:
            continue

        run_simulation(epochs, 
                       graph_options,
                       p_transmission, 
                       p_quarantine_exposed,
                       p_quarantine_infected,
                       exposure_days,
                       recovery_days,
                       num_clusters,
                       nodes_per_cluster,
                       inter_cluster_edges,
                       initial_infected,
                       plot=True)
        time.sleep(sleep_time)

def main():
    parser = argparse.ArgumentParser(description="SEIR+Q Disease Spread Simulation on Social Networks")
    parser.add_argument('--graph_type', type=str, default='SMALL_WORLD', choices=[g.name for g in GraphType], help='Type of network graph')
    parser.add_argument('--num_nodes', type=int, default=256, help='Number of nodes in the network')
    parser.add_argument('--avg_num_edges', type=int, default=8, help='Average degree/inter-cluster edges')
    parser.add_argument('--initial_infected', type=int, default=4, help='Initial number of infected individuals')
    parser.add_argument('--epochs', type=int, default=80, help='Number of simulation steps')
    parser.add_argument('--p_transmission', type=float, default=0.2, help='Transmission probability per contact')
    parser.add_argument('--p_qE', type=float, default=0.05, help='Quarantine probability for exposed')
    parser.add_argument('--p_qI', type=float, default=0.4, help='Quarantine probability for infected')
    parser.add_argument('--exposure_days', type=int, default=3, help='Days in exposed state before infectious')
    parser.add_argument('--recovery_days', type=int, default=14, help='Days in infectious state before recovery')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots')
    args = parser.parse_args()
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    graph_options = GraphOptions(GraphType[args.graph_type])
    run_simulation(
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
        do_plot=True,
        save_dir=args.save_dir
    )
if __name__ == "__main__":
    main()