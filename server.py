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

from tqdm import tqdm

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
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    prediction_distribution = torch.zeros(10)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(testloader, desc="Server-side Testing"):
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get predictions
            _, predicted = outputs.max(1)

            # Update prediction distribution
            for pred in predicted:
                prediction_distribution[pred] += 1

            # Update class-wise accuracy
            for label, prediction in zip(labels, predicted):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

            running_loss += loss.item()

    # Calculate metrics
    overall_accuracy = class_correct.sum() / class_total.sum() * 100
    avg_loss = running_loss / len(testloader)

    # Print detailed metrics
    print("\n" + "=" * 50)
    print("SERVER-SIDE EVALUATION METRICS")
    print("=" * 50)
    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")

    print("\nClass-wise Performance:")
    print("-" * 30)
    for i in range(10):
        accuracy = (
            (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
        )
        print(
            f"Class {i:2d}: {class_total[i]:4.0f} samples, {class_correct[i]:4.0f} correct, "
            f"Accuracy: {accuracy:6.2f}%"
        )

    print("\nModel Prediction Distribution:")
    print("-" * 30)
    total_predictions = prediction_distribution.sum()
    for i in range(10):
        percentage = (prediction_distribution[i] / total_predictions) * 100
        print(
            f"Class {i:2d}: {prediction_distribution[i]:4.0f} predictions ({percentage:5.2f}%)"
        )

    # Create metrics dictionary for saving
    class_accuracies = {}
    for i in range(10):
        acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
        class_accuracies[f"class_{i}_acc"] = float(acc)

    metrics = {
        "accuracy": float(overall_accuracy),
        "loss": float(avg_loss),
        "prediction_distribution": prediction_distribution.tolist(),
        **class_accuracies,
    }

    return avg_loss, overall_accuracy, metrics


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
            loss, accuracy, metrics = test(
                model, testloader, device=torch.device("cpu")
            )

            print("\n" + "=" * 50)
            print(f"ROUND {server_round} COMPLETE")
            print("=" * 50)

            # Save the model and metrics
            save_path = self.save_dir / f"model_round_{server_round}.pt"
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": parameters_dict,
                    "metrics": metrics,
                },
                save_path,
            )
            print(f"\nSaved aggregated model for round {server_round} to {save_path}")

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
            print(f"Saved latest model to {latest_path}")

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
