"""
SEIR+Q Disease Spread Simulation Core Logic

This module implements the core simulation engine for modeling disease transmission
through social networks with quarantine interventions. It provides:

- Network generation for various social network topologies
- SEIR+Q (Susceptible-Exposed-Infectious-Recovered + Quarantine) dynamics
- Simulation state tracking and history recording
- Epidemiological parameter management

The simulation models how diseases spread through social contacts, with the ability
to quarantine exposed and infected individuals to reduce transmission.
"""

from dataclasses import dataclass
import networkx as nx
import random
from enum import Enum
from typing import Optional


class GraphType(Enum):
    """Available network topology types for the simulation."""
    BINOMIAL = 1  # Erdős-Rényi random graph
    SMALL_WORLD = 2  # Watts-Strogatz small-world network
    PREFERENTIAL_ATTACHMENT = 3  # Barabási-Albert scale-free network
    STOCHASTIC_BLOCK = 4  # Stochastic block model (community structure)
    GEOMETRIC = 5  # Random geometric graph
    POWERLAW_CLUSTER = 6  # Power-law cluster graph


@dataclass
class SimulationParams:
    """
    Parameters for the SEIR+Q disease spread simulation.

    Attributes:
        graph_type: Type of social network topology to use
        num_nodes: Number of individuals in the network
        avg_num_edges: Average degree per node (varies by graph type)
        initial_infected: Number of initially infected individuals
        epochs: Number of simulation time steps (days)
        p_transmission: Probability of disease transmission per contact per day
        p_qE: Probability of quarantining an exposed individual per day
        p_qI: Probability of quarantining an infected individual per day
        exposure_days: Days in exposed state before becoming infectious
        recovery_days: Days in infectious state before recovery
    """
    graph_type: GraphType
    num_nodes: int = 256
    avg_num_edges: int = 8
    initial_infected: int = 4
    epochs: int = 80
    p_transmission: float = 0.2
    p_qE: float = 0.05
    p_qI: float = 0.4
    exposure_days: int = 3
    recovery_days: int = 14
    plot_population: bool = False
    plot_animation: bool = False
    save_dir: Optional[str] = None

def build_graph(graph_type, num_nodes, avg_num_edges, initial_infected):
    """Builds a social network graph of the specified type and initializes SEIR states."""
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
    s_count, e_count, i_count, qe_count, qi_count, r_count = 0, 0, 0, 0, 0, 0
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


def simulate_step(G, status_counts, p_transmission=0.05, p_quarantine_exposed=0.1,
                  p_quarantine_infected=0.5, exposure_days=3, recovery_days=10):
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
            if G.nodes[node]["quarantined"]:  # Doesn't infect other nodes
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

    S, E, I, QE, QI, R = get_status_counts(G)
    status_counts['S'].append(S)
    status_counts['E'].append(E)
    status_counts['I'].append(I)
    status_counts['QE'].append(QE)
    status_counts['QI'].append(QI)
    status_counts['R'].append(R)
    return status_counts

def run_simulation(params: SimulationParams,
                   record_history=False):
    """Runs the SEIR+Q simulation and returns the status counts."""
    G = build_graph(params.graph_type, params.num_nodes, params.avg_num_edges, params.initial_infected)

    # Counts of each status at each epoch, for plotting.
    status_counts = {k: [] for k in ["S", "E", "I", "QE", "QI", "R"]}
    history = [] if record_history else None

    for _ in range(params.epochs):
        if history is not None:
            history.append({n: G.nodes[n]["status"] for n in G.nodes})
        status_counts = simulate_step(G,
                                      status_counts,
                                      params.p_transmission,
                                      params.p_qE,
                                      params.p_qI,
                                      params.exposure_days,
                                      params.recovery_days)
    return status_counts, history, G
