# SEIR+Q Disease Spread Simulation on Social Networks

A comprehensive epidemiological simulation that models disease transmission through social networks with quarantine interventions. 

## Key Features

- **SEIR+Q Dynamics**: Models Susceptible-Exposed-Infectious-Recovered states with quarantine interventions
- **Multiple Network Topologies**: Small-world, scale-free, geometric, and other realistic social network structures
- **Advanced Visualizations**: Population-time plots, animated network evolution
- **Modular Architecture**: Clean, extensible codebase suitable for research and teaching

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/grimpala/DiseaseSpread
cd DiseaseSpread

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run a basic simulation with small-world network
python seir_simulation.py --graph_type SMALL_WORLD --p_transmission 0.25 --plot_population

# Create an animated visualization
python seir_simulation.py --graph_type SMALL_WORLD --p_transmission 0.25 --plot_animation

# Save visualizations locally
python seir_simulation.py --graph_type SMALL_WORLD --p_transmission 0.25 --plot_animation --plot_population --save_dir results/
```

## More Usage Examples

### 1. Full parameter customization
```bash
# Random geometric graph network with radius 0.1, moderate transmission and moderate quarantining
python seir_simulation.py --graph_type GEOMETRIC --p_transmission 0.25 --num_nodes 512 --avg_num_edges 8 --initial_infected 4 --epochs 120 --p_qE 0.01 --p_qI 0.3 --exposure_days 2 --recovery_days 7 --plot_animation --plot_population --save_dir results/
```

### 2. Animated Network Visualization
```bash
# Barabási-Albert scale-free network with animation
python main.py --graph_type PREFERENTIAL_ATTACHMENT --nodes 128 --edges 4 \
               --epochs 80 --plot_animation
```

## ⚙Parameter Reference

### Network Parameters
| Parameter         | Description                                        | Default | Range                                                                                                     |
|-------------------|----------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------|
| `--graph_type`    | Network topology type                              | `SMALL_WORLD` | `BINOMIAL`, `SMALL_WORLD`, `PREFERENTIAL_ATTACHMENT`, `STOCHASTIC_BLOCK`, `GEOMETRIC`, `POWERLAW_CLUSTER` |
| `--num_nodes`     | Number of individuals in network                   | `256` | `8-4096`                                                                                                  |
| `--avg_num_edges` | Average degree per node (for relevant graph types) | `8` | `4-64`                                                                                                    |

### Disease Parameters
| Parameter | Description | Default | Range     |
|-----------|-------------|---------|-----------|
| `--initial_infected` | Initial infected individuals | `4` | `0-16`    |
| `--epochs` | Simulation time steps (days) | `80` | `20-250`  |
| `--p_transmission` | Transmission probability per contact per day | `0.2` | `0.0-1.0` |
| `--exposure_days` | Days in exposed state before infectious | `3` | `0-10`    |
| `--recovery_days` | Days in infectious state before recovery | `14` | `2-30`    |

### Quarantine Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--p_qE` | Quarantine probability for exposed per day | `0.05` | `0.0-1.0` |
| `--p_qI` | Quarantine probability for infected per day | `0.4` | `0.0-1.0` |

### Output Options
| Parameter           | Description                                        | Default |
|---------------------|----------------------------------------------------|---------|
| `--save_dir`        | Directory to save plots and animations             | `None`  |
| `--plot_animation`  | Flag guarding animation visualization type         | `False` |
| `--plot_population` | Flag guarding static population visualization type | `False` |

## Project Structure

```
DiseaseSpread/
├── seir_simulation.py              # Visualization and simulation logic
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technical Details

### Network Models
- **Small-World (Watts-Strogatz)**: High clustering, short path lengths
- **Scale-Free (Barabási-Albert)**: Power-law degree distribution
- **Random (Erdős-Rényi)**: Poisson degree distribution
- **Geometric**: Spatial proximity-based connections
- **Power-Law Cluster**: Scale-free with clustering

### SEIR+Q States
- **S (Susceptible)**: Never infected, can become exposed
- **E (Exposed)**: Infected but not yet infectious
- **I (Infectious)**: Can transmit disease to susceptible contacts
- **R (Recovered)**: Immune, no longer infectious
- **QE/QI**: Quarantined versions of E/I states

### Visualization Features
- **Epidemic Curves**: Population-against-time charts showing state transitions
- **Network Animations**: Real-time visualization of disease spread

## Citation

If you use this code in your research, please cite:

```bibtex
@software{seir_network_simulation,
  title={SEIR+Q Disease Spread Simulation on Social Networks},
  author={Jacob Katzeff},
  year={2025},
  url={https://github.com/grimpala/DiseaseSpread}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.