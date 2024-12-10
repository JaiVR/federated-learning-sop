import argparse
from typing import List, Tuple
import os
from pathlib import Path

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.client_proxy import ClientProxy
import torch
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

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
    default=1,
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


def weighted_average(metrics: List[Tuple[ClientProxy, Metrics]]) -> Metrics:
    """Average the accuracy metric sent by clients in evaluate stage."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return configuration with better training parameters."""
    config = {
        "epochs": 5,  # Increase epochs
        "batch_size": 32,  # Slightly larger batch size
        "learning_rate": 0.0001,  # Controlled learning rate
    }
    return config


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def test(model: nn.Module, testloader: DataLoader, device: torch.device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Add class-wise tracking
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data["img"].to(device), data["label"].to(device)

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class-wise accuracy
            for label, prediction in zip(labels, predicted):
                class_correct[label] += (label == prediction).item()
                class_total[label] += 1

            running_loss += loss.item()

    accuracy = 100 * correct / total

    # Calculate and print class-wise accuracy
    print("\nServer-side Class-wise Accuracy:")
    class_accuracy = {}
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracy[f"class_{i}"] = class_acc
            print(f"Class {i}: {class_acc:.2f}%")

    return running_loss / len(testloader), accuracy, class_accuracy


def prepare_test_dataset():
    """Prepare the test dataset for server-side evaluation."""
    from flwr_datasets import FederatedDataset

    img_key = "img"
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the dataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    # Load and transform the test set
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 50})
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    return testset


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
            parameters_dict = dict(
                zip(
                    model.state_dict().keys(),
                    fl.common.parameters_to_ndarrays(aggregated_parameters),
                )
            )

            # Evaluate aggregated model on the testset
            testset = prepare_test_dataset()
            testloader = DataLoader(testset, batch_size=64, num_workers=0)
            loss, accuracy, class_accuracy = test(
                model, testloader, device=torch.device("cpu")
            )

            print(f"\nRound {server_round}:")
            print(f"Overall Test Accuracy: {accuracy:.4f}")

            # Save metrics including class-wise accuracy
            metrics = {"accuracy": accuracy, "loss": loss, **class_accuracy}

            # Save the model with updated metrics
            save_path = self.save_dir / f"model_round_{server_round}.pt"
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": parameters_dict,
                    "metrics": metrics,
                },
                save_path,
            )

            # Save as latest model for inference
            latest_path = self.save_dir / "model_latest.pt"
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": parameters_dict,
                    "metrics": metrics,
                },
                latest_path,
            )

            return aggregated_parameters, metrics

        return aggregated_parameters, aggregated_metrics


def get_parameters(model):
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


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
