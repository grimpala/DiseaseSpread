import argparse
import os
from simulation import SimulationParams, GraphType, run_simulation
from visualization import plot_statuses, animate_network

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
        return argparse.ArgumentTypeError(val_type, f'Received {val_type} of type {type(val_type)};'
                                                    f' expected {self.expected_type}')
    def range_exception(self, value):
        if self.low is not None and self.high is not None:
            return argparse.ArgumentError(value, f"Must be a {self.expected_type} in the range"
                                                 f" [{self.low}, {self.high}]")
        elif self.low is not None:
            return argparse.ArgumentError(value, f"Must be an {self.expected_type} >= {self.low}")
        elif self.high is not None:
            return argparse.ArgumentError(value, f"Must be an {self.expected_type} <= {self.high}")
        else:
            return argparse.ArgumentError(value, "Error with argument in NumRange")


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SEIR+Q Disease Spread Simulation on Social Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Basic simulation with small-world network
      python main.py --graph_type SMALL_WORLD --num_nodes 256 --p_transmission 0.15

      # Save animated visualization with scale-free network
      python main.py --graph_type PREFERENTIAL_ATTACHMENT --plot_animation --save_dir results/
    """
    )
    parser.add_argument('--graph_type', type=str,
                        default='SMALL_WORLD', choices=[g.name for g in GraphType], help='Type of network graph')
    parser.add_argument('--num_nodes', type=NumRange(2**3, 2**12, var_type=int),
                        default=256, help='Number of nodes in the network')
    parser.add_argument('--avg_num_edges', type=NumRange(2**2, 2**6, var_type=int),
                        default=8, help='Average degree/inter-cluster edges')
    parser.add_argument('--initial_infected', type=NumRange(high=2**4, var_type=int),
                        default=4, help='Initial number of infected individuals')
    parser.add_argument('--epochs', type=NumRange(20, 251, var_type=int),
                        default=80, help='Number of simulation steps')
    parser.add_argument('--p_transmission', type=NumRange(0.0, 1.0, var_type=float),
                        default=0.2, help='Transmission probability per contact')
    parser.add_argument('--p_qE', type=NumRange(0.0, 1.0, var_type=float),
                        default=0.05, help='Quarantine probability for exposed')
    parser.add_argument('--p_qI', type=NumRange(0.0, 1.0, var_type=float),
                        default=0.4, help='Quarantine probability for infected')
    parser.add_argument('--exposure_days', type=NumRange(high=10, var_type=int),
                        default=3, help='Days in exposed state before infectious')
    parser.add_argument('--recovery_days', type=NumRange(2,30, var_type=int),
                        default=14, help='Days in infectious state before recovery')
    parser.add_argument('--plot_population', action='store_true', help='Plot population against time')
    parser.add_argument('--plot_animation', action='store_true',
                        help='Create an animated network visualization')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots')
    return parser

def run_animation(params: SimulationParams) -> None:
    """Run animated network visualization."""
    status_counts, history, G = run_simulation(params, record_history=True)

    if history is not None and params.plot_animation:
        animate_network(
            G, history,
            interval=200,
            save_dir=params.save_dir
        )
    else:
        print("Error: No history recorded for animation.")

def run_standard_simulation(params: SimulationParams) -> None:
    status_counts, _, G = run_simulation(params)

    # Create parameter string for plot titles
    param_str = (f"p_trans={params.p_transmission:.2f}, p_qE={params.p_qE:.2f}, p_qI={params.p_qI:.2f},\n"
                    f"exp_days={params.exposure_days}, rec_days={params.recovery_days}, nodes={params.num_nodes}, edges={params.avg_num_edges}")

    # Generate plots
    plot_path = os.path.join(params.save_dir, "epidemic_curve.png") if params.save_dir else None
    plot_statuses(status_counts, plot_title=param_str, save_dir=plot_path)

def main() -> None:
    """Main entry point for the SEIR+Q simulation."""
    args = setup_argument_parser().parse_args()
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create simulation parameters
    params = SimulationParams(
        graph_type=GraphType[args.graph_type],
        num_nodes=args.num_nodes,
        avg_num_edges=args.avg_num_edges,
        initial_infected=args.initial_infected,
        epochs=args.epochs,
        p_transmission=args.p_transmission,
        p_qE=args.p_qE,
        p_qI=args.p_qI,
        exposure_days=args.exposure_days,
        recovery_days=args.recovery_days,
        plot_population=args.plot_population,
        plot_animation=args.plot_animation,
        save_dir=args.save_dir

    )
    if args.plot_animation:
        run_animation(params)
    elif args.plot_population:
        run_standard_simulation(params)

if __name__ == "__main__":
    main()