import argparse
from typing import List, Tuple
import os
from pathlib import Path

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.client_proxy import ClientProxy
import torch
from torchvision.models import mobilenet_v3_small

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np

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
    default=10,
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


# Instantiate model
model = mobilenet_v3_small(num_classes=10)


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
            parameters_dict = dict(zip(model.state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters)))

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


def get_parameters(model):
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def load_initial_parameters(model, checkpoint_path="saved_models/model_latest.pt"):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        parameters_dict = checkpoint['model_state_dict']
        
        # Convert all numpy arrays in parameters_dict to torch.Tensor
        if isinstance(parameters_dict, list):
            parameters_dict = dict(zip(model.state_dict().keys(), [torch.tensor(param) for param in parameters_dict]))
        
        # Ensure all elements in parameters_dict are tensors
        for key, value in parameters_dict.items():
            if isinstance(value, np.ndarray):  # Convert if necessary
                parameters_dict[key] = torch.tensor(value)
        
        model.load_state_dict(parameters_dict)
        print("Existing model parameters loaded!")
        
        # Return parameters as a list of NumPy arrays for Flower
        return fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
    except FileNotFoundError:
        print("No existing model parameters found!")
        return False


def main():
    args = parser.parse_args()
    print(args)

    parameters = load_initial_parameters(model)    

    # Define strategy with model saving
    if parameters:
        strategy = SaveModelStrategy(
            save_dir=args.save_dir,
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
        )
    else:
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
