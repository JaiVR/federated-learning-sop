import argparse
from typing import List, Tuple
import os
from pathlib import Path

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.client_proxy import ClientProxy
import torch

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="saved_models",
    help="Directory to save the trained models (default: 'saved_models')",
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Average the accuracy metric sent by clients in evaluate stage."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return configuration with static batch size and epochs."""
    config = {
        "epochs": 5,
        "batch_size": 16,
    }
    return config


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that saves the model after training."""

    def __init__(
        self,
        save_dir: str = "saved_models",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> tuple[Parameters, dict]:
        """Aggregate model weights and save the model after each round."""
        # Aggregate weights using parent class method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert parameters to PyTorch state_dict
            parameters_dict = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save the model
            save_path = self.save_dir / f"model_round_{server_round}.pt"
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": parameters_dict,
                    "metrics": aggregated_metrics,
                },
                save_path,
            )
            print(f"Saved aggregated model for round {server_round} to {save_path}")

            # Save as latest model for inference
            latest_path = self.save_dir / "model_latest.pt"
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": parameters_dict,
                    "metrics": aggregated_metrics,
                },
                latest_path,
            )
            print(f"Saved latest model to {latest_path}")

        return aggregated_parameters, aggregated_metrics


def main():
    args = parser.parse_args()
    print(args)

    # Define strategy with model saving
    strategy = SaveModelStrategy(
        save_dir=args.save_dir,
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
