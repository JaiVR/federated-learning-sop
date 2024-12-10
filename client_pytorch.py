import argparse
import warnings
from collections import OrderedDict
import os
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints",
    help="Directory to save client checkpoints",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 20

CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def train(net, trainloader, optimizer, epochs, device, **kwargs):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100.0 * correct / total
        print(f"\nEpoch {epoch + 1}")
        print(f"Training Loss: {running_loss/len(trainloader):.3f}")
        print(f"Training Accuracy: {epoch_acc:.2f}%")

        if epoch == epochs - 1:
            print("\nFinal Epoch Prediction Distribution:")
            print("-" * 60)
            print(f"{'Class':<20} | {'Predictions':>12} | {'Percentage':>10}")
            print("-" * 60)

            pred_dist = torch.zeros(10, device=device)
            with torch.no_grad():
                for batch in trainloader:
                    outputs = net(batch["img"].to(device))
                    _, preds = outputs.max(1)
                    for i in range(10):
                        pred_dist[i] += (preds == i).sum().item()

            total_preds = pred_dist.sum()
            for i in range(10):
                percentage = (pred_dist[i] / total_preds) * 100
                class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
                print(f"{class_name:<20} | {pred_dist[i]:12.0f} | {percentage:9.2f}%")


def test(net, testloader, device, get_class_acc=False):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    net.eval()

    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    prediction_distribution = torch.zeros(10)

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            batch = list(batch.values())
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                class_correct[label] += (label == prediction).item()
                class_total[label] += 1

            for pred in predicted:
                prediction_distribution[pred] += 1

    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)

    if get_class_acc:
        print("\n" + "=" * 60)
        print("CLIENT-SIDE EVALUATION METRICS")
        print("=" * 60)
        print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
        print(f"Average Loss: {avg_loss:.4f}")

        print("\nClass-wise Performance:")
        print("-" * 60)
        print(f"{'Class':<20} | {'Samples':>8} | {'Correct':>8} | {'Accuracy':>10}")
        print("-" * 60)

        class_accuracy = {}
        for i in range(10):
            if class_total[i] > 0:
                acc = float(class_correct[i] / class_total[i])
                class_accuracy[f"class_{i}_{CIFAR10_LABELS[i]}_acc"] = acc
                class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
                print(
                    f"{class_name:<20} | {class_total[i]:8.0f} | {class_correct[i]:8.0f} | {acc*100:9.2f}%"
                )
            else:
                class_accuracy[f"class_{i}_{CIFAR10_LABELS[i]}_acc"] = 0.0
                class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
                print(f"{class_name:<20} | {0:8.0f} | {0:8.0f} | {0:9.2f}%")

        print("\nModel Prediction Distribution:")
        print("-" * 60)
        print(f"{'Class':<20} | {'Predictions':>12} | {'Percentage':>10}")
        print("-" * 60)
        total_predictions = prediction_distribution.sum()
        for i in range(10):
            percentage = (prediction_distribution[i] / total_predictions) * 100
            class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
            print(
                f"{class_name:<20} | {prediction_distribution[i]:12.0f} | {percentage:9.2f}%"
            )

        return avg_loss, accuracy, class_accuracy

    return avg_loss, accuracy


# Dataset preparation function remains the same
def prepare_dataset():
    """Get CIFAR-10 and return client partitions and global testset."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    img_key = "img"
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    # random_seed = np.random.randint(0, 10000)  # Adjust range if needed
    # print(f"Using random seed: {random_seed}")
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=1000)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    return trainsets, validsets, testset


class FlowerClient(fl.client.NumPyClient):
    """Enhanced FlowerClient with improved model management and checkpointing."""

    def __init__(self, trainset, valset, checkpoint_dir=None, client_id=None):
        self.trainset = trainset
        self.valset = valset
        self.checkpoint_dir = checkpoint_dir
        self.client_id = client_id

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model with proper weights
        self.model = mobilenet_v3_small(
            pretrained=True
        )  # Start with pretrained weights
        # Modify the classifier for CIFAR-10
        self.model.classifier[-1] = nn.Linear(in_features=1024, out_features=10)
        self.model.to(self.device)

        # Use a more stable optimizer configuration
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.0001, weight_decay=0.01
        )

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.1, patience=2, verbose=True
        )

    def set_parameters(self, params):
        """Set model parameters from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        print(f"Client {self.client_id}: Starting fit()")
        self.set_parameters(parameters)

        # Adjust batch norm momentum for better training
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = 0.01

        trainloader = DataLoader(
            self.trainset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,  # Add this for better performance
        )

        # Print class distribution
        class_counts = torch.zeros(10)
        for item in self.trainset:
            label = item["label"]
            class_counts[label] += 1

        # In the fit method of FlowerClient
        print("\nClass distribution in training set:")
        print("-" * 60)
        print(f"{'Class':<20} | {'Samples':>8} | {'Percentage':>10}")
        print("-" * 60)
        total_samples = class_counts.sum()
        for class_idx, count in enumerate(class_counts):
            percentage = (count / total_samples) * 100
            class_name = f"Class {class_idx} ({CIFAR10_LABELS[class_idx]})"
            print(f"{class_name:<20} | {int(count):8d} | {percentage:9.2f}%")

        print("\nClass distribution in training set:")
        for class_idx, count in enumerate(class_counts):
            print(f"Class {class_idx}: {int(count)} samples")

        batch_size = config["batch_size"]
        epochs = config["epochs"]

        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        train(
            net=self.model,
            trainloader=trainloader,
            optimizer=self.optimizer,
            epochs=epochs,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
            client_id=self.client_id,
        )

        # Update learning rate based on training performance
        loss, accuracy = test(self.model, trainloader, self.device)
        self.scheduler.step(accuracy)

        print(
            f"Client {self.client_id}: Local training accuracy: {accuracy * 100:.2f}%"
        )

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation dataset."""
        print(f"Client {self.client_id}: Starting evaluate()")
        self.set_parameters(parameters)

        # Add this code to check validation set distribution
        class_counts = torch.zeros(10)
        for item in self.valset:
            label = item["label"]  # Get single label
            class_counts[label] += 1

        print("\nClass distribution in validation set:")
        for class_idx, count in enumerate(class_counts):
            print(f"Class {class_idx}: {int(count)} samples")

        valloader = DataLoader(self.valset, batch_size=64, num_workers=0)

        loss, accuracy, class_accuracy = test(
            self.model, valloader, self.device, get_class_acc=True
        )

        print(f"\nClient {self.client_id} Class-wise Accuracy:")
        for class_id, acc in class_accuracy.items():
            print(f"{class_id}: {acc*100:.2f}%")

        checkpoint_dir = self.checkpoint_dir
        client_id = self.client_id
        if checkpoint_dir is not None and client_id is not None:
            checkpoint_path = Path(checkpoint_dir) / f"client_{client_id}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "class_accuracy": class_accuracy,
                },
                checkpoint_path,
            )

        # Include class-wise accuracy in the returned metrics
        return (
            float(loss),
            len(valloader.dataset),
            {"accuracy": float(accuracy), **class_accuracy},
        )


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # Create checkpoint directory if specified
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Download dataset and partition it
    trainsets, valsets, _ = prepare_dataset()

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid],
            valset=valsets[args.cid],
            checkpoint_dir=args.checkpoint_dir,
            client_id=args.cid,
        ).to_client(),
    )


if __name__ == "__main__":
    main()
