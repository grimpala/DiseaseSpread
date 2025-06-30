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

def build_graph(graph_options, num_clusters, nodes_per_cluster, inter_cluster_edges, initial_infected):
    # Initialize Graph

    graph_type = graph_options.graph_type
    if graph_type == GraphType.BINOMIAL:
        G = nx.erdos_renyi_graph(n=nodes_per_cluster, p=10/n)
    elif graph_type == GraphType.SMALL_WORLD:
        G = nx.watts_strogatz_graph(n=nodes_per_cluster, k=inter_cluster_edges, p=0.1)
    elif graph_type == GraphType.PREFERENTIAL_ATTACHMENT:
        G = nx.barabasi_albert_graph(n=nodes_per_cluster, k=inter_cluster_edges)
    elif graph_type == GraphType.STOCHASTIC_BLOCK:
        # TODO, default to small world
        G = nx.watts_strogatz_graph(n=nodes_per_cluster, k=inter_cluster_edges, p=0.1)
    elif graph_type == GraphType.GEOMETRIC:
        G = nx.random_geometric_graph(n=nodes_per_cluster, radius=0.1)
    elif graph_type == GraphType.POWERLAW_CLUSTER:
        G = nx.powerlaw_cluster_graph(n=nodes_per_cluster, m=inter_cluster_edges, p=0.3)
    # G = nx.Graph()
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    cluster_nodes = []

    # Initialize states in Graph.
    for node in G.nodes():
        G.nodes[node]["status"] = "S"
        G.nodes[node]["quarantined"] = False

    # Add one random infected node
    for infected_node in random.sample(list(G.nodes()), initial_infected):
        
        G.nodes[infected_node]["status"] = "I" 
        G.nodes[infected_node]["infected_days"] = 0

    return G, cluster_nodes

def get_status_counts(G):
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

def plot_statuses(status_counts, p_transmission, p_quarantine_exposed, p_quarantine_infected, num_clusters, exposure_days, recovery_days, nodes_per_cluster, inter_cluster_edges, fractional=False):
    population = num_clusters * nodes_per_cluster
    
    S = np.array(status_counts['S'])
    E = np.array(status_counts['E'])
    I = np.array(status_counts['I'])
    QE = np.array(status_counts['QE'])
    QI = np.array(status_counts['QI'])
    R = np.array(status_counts['R'])

    # Normalize to fractions
    S_frac = S / population
    E_frac = E / population
    I_frac = I / population
    QE_frac = QE / population
    QI_frac = QI / population
    R_frac = R / population
    
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
    plt.title(f'p_transmission={p_transmission:.2f}, p_quarantine_exposed={p_quarantine_exposed:.1f}, p_quarantine_infected={p_quarantine_infected:.1f} exposure_days={exposure_days}, recovery_days={recovery_days}, num_clusters={num_clusters}, nodes_per_cluster={nodes_per_cluster}, inter_cluster_edges={inter_cluster_edges}')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_simulation(epochs,
                   graph_options,
                   p_transmission, 
                   p_quarantine_exposed, 
                   p_quarantine_infected, 
                   exposure_days, 
                   recovery_days, 
                   num_clusters, 
                   nodes_per_cluster, 
                   inter_cluster_edges, 
                   initial_infected=1,
                   prebuilt_graph=None, 
                   plot=False):
    if prebuilt_graph is not None:
        G = prebuilt_graph
    else:
        G, _ = build_graph(graph_options, num_clusters, nodes_per_cluster, inter_cluster_edges, initial_infected=initial_infected)
    
    population = num_clusters * nodes_per_cluster
        
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
    if plot:
        display.clear_output(wait=True)
        plot_statuses(status_counts,
                      p_transmission,
                      p_quarantine_exposed,
                      p_quarantine_infected,
                      num_clusters,
                      exposure_days,
                      recovery_days,
                      nodes_per_cluster,
                      inter_cluster_edges)

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

full_param_sweep = {
    "p_transmission": [0.4, 0.2, 0.1, 0.05],
    "p_quarantine_exposed": [0.0, 0.1, 0.2, 0.4],
    "p_quarantine_infected": [0.3, 0.7, 1.0],
    "exposure_days": [2,3],
    "recovery_days": [7,14],
    "num_clusters": [4,8],
    "nodes_per_cluster": [64,128],
    "inter_cluster_edges": [8,32],
    "initial_infected": [4],
}

transmission_dynamic_params = {
}

intervention_policy_params = {
    "p_transmission": [0.1],
    "p_quarantine_exposed": [0.0, 0.05, 0.1, 0.2, 0.4],
    "p_quarantine_infected": [0.0, 0.3, 0.7, 0.9],
    "exposure_days": [3],
    "recovery_days": [14],
    "num_clusters": [8],
    "nodes_per_cluster": [128],
    "inter_cluster_edges": [32],
    "initial_infected": [4],
}

network_structure_params = {
}

baseline_params = {
    "p_transmission": [0.2],
    "p_quarantine_exposed": [0.05],
    "p_quarantine_infected": [0.4],
    "exposure_days": [3],
    "recovery_days": [14],
    "num_clusters": [8],
    "nodes_per_cluster": [128],
    "inter_cluster_edges": [8],
    "initial_infected": [4],
}

graph_options = GraphOptions(GraphType.SMALL_WORLD)

print(num_param_combinations(full_param_sweep))
for _ in range(1):
    plot_variable_params(baseline_params, GraphOptions(GraphType.POWERLAW_CLUSTER), sleep_time=1)