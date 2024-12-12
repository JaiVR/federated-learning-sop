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

# Argument parser setup to configure client parameters via command line
parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",  # Default gRPC server address
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,  # Client ID is required
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints",  # Directory to store checkpoints
    help="Directory to save client checkpoints",
)

# Ignore warnings of the UserWarning category
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
NUM_CLIENTS = 20  # Total number of clients in the federated learning setup

# CIFAR-10 class labels for reference (mapping class index to class name)
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
    """
    Train the model on the CIFAR-10 training set.

    Args:
        net: The neural network model to train (e.g., MobileNetV3).
        trainloader: DataLoader object providing training batches.
        optimizer: Optimizer used to update the model parameters (e.g., SGD).
        epochs: Number of epochs to train the model.
        device: The device on which the model is trained (CPU or GPU).
        **kwargs: Additional keyword arguments (unused in this function).
    """
    # CrossEntropyLoss with label smoothing to improve generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Set the model to training mode (affects layers like dropout, batchnorm)
    net.train()

    # Iterate over epochs for training
    for epoch in range(epochs):
        running_loss = 0.0  # Variable to accumulate the loss over the epoch
        correct = 0  # Number of correct predictions
        total = 0  # Total number of samples processed

        # Iterate through the training data in batches
        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["img"].to(device)  # Move images to the correct device
            labels = batch["label"].to(device)  # Move labels to the correct device

            optimizer.zero_grad()  # Zero the gradients before backpropagation
            outputs = net(images)  # Forward pass: get model predictions
            loss = criterion(
                outputs, labels
            )  # Compute the loss between predictions and true labels
            loss.backward()  # Backpropagate the loss to compute gradients

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()  # Update the model parameters using the optimizer

            # Accumulate loss for the epoch
            running_loss += loss.item()

            # Get the predictions from the output probabilities (taking the class with max probability)
            _, predicted = outputs.max(1)

            # Count the number of correct predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Compute accuracy for this epoch
        epoch_acc = 100.0 * correct / total
        print(f"\nEpoch {epoch + 1}")
        print(f"Training Loss: {running_loss/len(trainloader):.3f}")
        print(f"Training Accuracy: {epoch_acc:.2f}%")

        # At the final epoch, print out the class-wise prediction distribution
        if epoch == epochs - 1:
            print("\nFinal Epoch Prediction Distribution:")
            print("-" * 60)
            print(f"{'Class':<20} | {'Predictions':>12} | {'Percentage':>10}")
            print("-" * 60)

            pred_dist = torch.zeros(
                10, device=device
            )  # Initialize tensor to track class-wise predictions

            # No gradient computation needed for prediction distribution
            with torch.no_grad():
                # Iterate over all batches in the trainloader to accumulate predictions
                for batch in trainloader:
                    outputs = net(batch["img"].to(device))  # Get model predictions
                    _, preds = outputs.max(1)  # Get predicted classes for each image
                    # Update the prediction distribution count for each class
                    for i in range(10):
                        pred_dist[i] += (preds == i).sum().item()

            total_preds = pred_dist.sum()  # Total number of predictions made
            for i in range(10):
                # Calculate and print the percentage of predictions for each class
                percentage = (pred_dist[i] / total_preds) * 100
                class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
                print(f"{class_name:<20} | {pred_dist[i]:12.0f} | {percentage:9.2f}%")


def test(net, testloader, device, get_class_acc=False):
    """
    Validate the model on the test set, calculate accuracy, loss, and optionally
    return class-wise accuracy and prediction distribution.

    Args:
        net: The trained neural network model.
        testloader: DataLoader providing the test data.
        device: The device to run the model on (CPU or GPU).
        get_class_acc: If True, returns class-wise accuracy and prediction distribution.

    Returns:
        avg_loss: The average loss over the test set.
        accuracy: Overall accuracy on the test set.
        class_accuracy (optional): A dictionary containing class-wise accuracy if `get_class_acc` is True.
    """
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification
    correct = 0  # Correct predictions
    total_loss = 0.0  # Accumulated loss

    # Arrays to store class-wise accuracy data and prediction distribution
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    prediction_distribution = torch.zeros(10)

    # Switch the model to evaluation mode (disables layers like dropout)
    net.eval()

    # Disable gradient calculation during testing (no backpropagation needed)
    with torch.no_grad():
        # Iterate over the test dataset in batches
        for batch in tqdm(testloader, desc="Testing"):
            batch = list(batch.values())  # Extract image and label from the batch
            images, labels = batch[0].to(device), batch[1].to(
                device
            )  # Move to device (CPU/GPU)
            outputs = net(images)  # Forward pass: Get model predictions
            total_loss += criterion(
                outputs, labels
            ).item()  # Compute and accumulate loss

            _, predicted = torch.max(
                outputs.data, 1
            )  # Get predicted class for each image
            correct += (predicted == labels).sum().item()  # Count correct predictions

            # Calculate class-wise accuracy by comparing each label and prediction
            for label, prediction in zip(labels, predicted):
                class_correct[label.item()] += (label == prediction).item()
                class_total[label.item()] += 1

            # Update the prediction distribution for each predicted class
            for pred in predicted:
                prediction_distribution[pred.item()] += 1

    # Overall accuracy and average loss for the test set
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)

    # If class-wise accuracy is required, print detailed metrics
    if get_class_acc:
        print("\n" + "=" * 60)
        print("CLIENT-SIDE EVALUATION METRICS")
        print("=" * 60)
        print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
        print(f"Average Loss: {avg_loss:.4f}")

        # Print class-wise performance metrics
        print("\nClass-wise Performance:")
        print("-" * 60)
        print(f"{'Class':<20} | {'Samples':>8} | {'Correct':>8} | {'Accuracy':>9}")
        print("-" * 60)

        class_accuracy = {}
        for i in range(10):
            total = float(class_total[i].item())  # Total samples in class i
            correct = float(class_correct[i].item())  # Correct predictions for class i
            acc = (
                (correct / total * 100) if total > 0 else 0.0
            )  # Class accuracy percentage
            class_accuracy[f"class_{i}_{CIFAR10_LABELS[i]}_acc"] = float(
                acc / 100
            )  # Store as decimal
            class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
            print(f"{class_name:<20} | {total:8.0f} | {correct:8.0f} | {acc:8.2f}%")

        # Print model's prediction distribution across all classes
        print("\nModel Prediction Distribution:")
        print("-" * 60)
        print(f"{'Class':<20} | {'Predictions':>12} | {'Percentage':>10}")
        print("-" * 60)
        total_predictions = float(
            prediction_distribution.sum().item()
        )  # Total number of predictions
        for i in range(10):
            pred_count = float(
                prediction_distribution[i].item()
            )  # Number of predictions for class i
            percentage = (
                pred_count / total_predictions * 100
            )  # Percentage of total predictions for class i
            class_name = f"Class {i} ({CIFAR10_LABELS[i]})"
            print(f"{class_name:<20} | {pred_count:12.0f} | {percentage:9.2f}%")

        # Return average loss, accuracy, and class-wise accuracy as a dictionary
        return float(avg_loss), float(accuracy), class_accuracy

    # Return average loss and accuracy if class-wise metrics are not requested
    return float(avg_loss), float(accuracy)


def prepare_dataset():
    """
    Prepare CIFAR-10 dataset with balanced IID partitions for federated learning.
    Each client gets a balanced number of samples per class (IID distribution).
    The training data is split into training and validation sets (80-20), and the test set is balanced across clients.

    Returns:
        trainsets: A list of training datasets for each client.
        validsets: A list of validation datasets for each client.
        balanced_testset: A balanced test dataset for evaluation.
    """
    # Initialize FederatedDataset with CIFAR-10 and specify partitioners for training set
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

    # Define image transformations (normalization and tensor conversion)
    img_key = "img"  # Key for the images in the dataset
    norm = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # Image normalization
    pytorch_transforms = Compose([ToTensor(), norm])  # Combine to Tensor and normalize

    # Function to apply transformations to each batch
    def apply_transforms(batch):
        batch[img_key] = [
            pytorch_transforms(img) for img in batch[img_key]
        ]  # Apply to each image in batch
        return batch

    trainsets = []  # List to store training datasets for each client
    validsets = []  # List to store validation datasets for each client

    # Calculate how many samples each client should have
    total_train_samples = 50000  # Total number of training samples in CIFAR-10
    samples_per_client = total_train_samples // NUM_CLIENTS  # Samples per client
    samples_per_class_per_client = samples_per_client // 10  # 10 classes in CIFAR-10

    print(f"\nDistribution Info:")
    print(f"Total clients: {NUM_CLIENTS}")
    print(f"Samples per client: {samples_per_client}")
    print(f"Samples per class per client: {samples_per_class_per_client}")

    # Loop over each client to create their partition of data
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(
            partition_id, "train"
        )  # Load the partition for each client

        # Initialize lists to hold the training and validation indices
        train_indices = []
        val_indices = []

        # Distribute data per class for each client
        for class_idx in range(10):
            class_indices = [
                i for i, item in enumerate(partition) if item["label"] == class_idx
            ]  # Indices of the current class

            # Ensure balanced data by selecting the required number of samples for each class
            if len(class_indices) >= samples_per_class_per_client:
                selected_indices = class_indices[
                    :samples_per_class_per_client
                ]  # Select the first samples
            else:
                # If there are fewer samples than needed, randomly select with replacement
                selected_indices = np.random.choice(
                    class_indices, size=samples_per_class_per_client, replace=True
                )

            # Split the selected indices into training (80%) and validation (20%) sets
            n_val = int(len(selected_indices) * 0.2)
            val_indices.extend(selected_indices[:n_val])
            train_indices.extend(selected_indices[n_val:])

        # Create the training and validation datasets for the current client
        train_partition = partition.select(train_indices)
        val_partition = partition.select(val_indices)

        # Apply the transformations to the datasets
        train_partition = train_partition.with_transform(apply_transforms)
        val_partition = val_partition.with_transform(apply_transforms)

        # Append to the lists for all clients
        trainsets.append(train_partition)
        validsets.append(val_partition)

    # For test set, ensure equal distribution of samples across all clients
    testset = fds.load_split("test")  # Load the test set
    samples_per_class_test = (
        1000 // NUM_CLIENTS
    )  # Each client gets an equal portion of test set

    # Distribute test samples across clients
    balanced_test_indices = []
    for class_idx in range(10):
        class_indices = [
            i for i, item in enumerate(testset) if item["label"] == class_idx
        ]  # Indices of the current class in the test set

        # Ensure each client gets the required number of samples for each class
        if len(class_indices) >= samples_per_class_test:
            selected_indices = class_indices[
                :samples_per_class_test
            ]  # Select the first samples
        else:
            selected_indices = np.random.choice(
                class_indices, size=samples_per_class_test, replace=True
            )  # Select with replacement if there aren't enough samples

        balanced_test_indices.extend(
            selected_indices
        )  # Collect indices for the balanced test set

    # Create the balanced test dataset
    balanced_testset = testset.select(balanced_test_indices)
    balanced_testset = balanced_testset.with_transform(
        apply_transforms
    )  # Apply transformations

    # Return the training, validation, and balanced test datasets
    return trainsets, validsets, balanced_testset


class FlowerClient(fl.client.NumPyClient):
    """Enhanced FlowerClient with improved model management and checkpointing."""

    def __init__(self, trainset, valset, checkpoint_dir=None, client_id=None):
        """
        Initialize the Flower client with local training and validation datasets,
        model, optimizer, and checkpointing directory.

        Args:
            trainset: Local training dataset.
            valset: Local validation dataset.
            checkpoint_dir: Directory to store model checkpoints.
            client_id: Unique identifier for the client.
        """
        self.trainset = trainset
        self.valset = valset
        self.checkpoint_dir = checkpoint_dir
        self.client_id = client_id

        # Determine the device for training (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the model with pretrained MobileNetV3 small
        self.model = mobilenet_v3_small(pretrained=True)
        # Modify the final classifier layer for CIFAR-10 classification (10 classes)
        self.model.classifier[-1] = nn.Linear(in_features=1024, out_features=10)
        self.model.to(self.device)  # Move the model to the specified device

        # Set up a stable optimizer configuration using AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.0001, weight_decay=0.01
        )

        # Learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.1, patience=2, verbose=True
        )

    def set_parameters(self, params):
        """
        Set model parameters from a list of NumPy ndarrays.

        Args:
            params: List of NumPy ndarrays representing model parameters.
        """
        # Convert parameters to torch tensors and load into the model
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """
        Get model parameters as a list of NumPy ndarrays.

        Args:
            config: Configuration dict passed by the server (not used here).
        """
        # Return the model's parameters as NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """
        Train the model on the local dataset.

        Args:
            parameters: List of model parameters from the server.
            config: Configuration dictionary containing training settings.
        """
        print(f"Client {self.client_id}: Starting fit()")
        self.set_parameters(parameters)

        # Adjust batch norm momentum for more stable training
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = 0.01

        # Prepare data loader for training
        trainloader = DataLoader(
            self.trainset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,  # Ensures faster data loading
        )

        # Print class distribution for the training set
        class_counts = torch.zeros(10)
        for item in self.trainset:
            label = item["label"]
            class_counts[label] += 1

        print("\nClass distribution in training set:")
        print("-" * 60)
        print(f"{'Class':<20} | {'Samples':>8} | {'Percentage':>10}")
        print("-" * 60)
        total_samples = class_counts.sum()
        for class_idx, count in enumerate(class_counts):
            percentage = (count / total_samples) * 100
            class_name = f"Class {class_idx} ({CIFAR10_LABELS[class_idx]})"
            print(f"{class_name:<20} | {int(count):8d} | {percentage:9.2f}%")

        batch_size = config["batch_size"]
        epochs = config["epochs"]

        # Additional training loader with batch size from config
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Start the training loop
        train(
            net=self.model,
            trainloader=trainloader,
            optimizer=self.optimizer,
            epochs=epochs,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
            client_id=self.client_id,
        )

        # Evaluate the model after training to get loss and accuracy
        loss, accuracy = test(self.model, trainloader, self.device)
        # Update the learning rate based on training performance (accuracy)
        self.scheduler.step(accuracy)

        print(
            f"Client {self.client_id}: Local training accuracy: {accuracy * 100:.2f}%"
        )

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local validation dataset.

        Args:
            parameters: List of model parameters from the server.
            config: Configuration dictionary passed by the server.
        """
        print(f"Client {self.client_id}: Starting evaluate()")
        self.set_parameters(parameters)

        # Print class distribution for the validation set
        class_counts = torch.zeros(10)
        for item in self.valset:
            label = item["label"]
            class_counts[label] += 1

        print("\nClass distribution in validation set:")
        for class_idx, count in enumerate(class_counts):
            print(f"Class {class_idx}: {int(count)} samples")

        # Prepare validation loader
        valloader = DataLoader(self.valset, batch_size=64, num_workers=0)

        # Test the model on the validation set
        loss, accuracy, class_accuracy = test(
            self.model, valloader, self.device, get_class_acc=True
        )

        print(f"\nClient {self.client_id} Class-wise Accuracy:")
        for class_id, acc in class_accuracy.items():
            print(f"{class_id}: {acc*100:.2f}%")

        # Save model checkpoint
        if self.checkpoint_dir is not None and self.client_id is not None:
            checkpoint_path = Path(self.checkpoint_dir) / f"client_{self.client_id}.pt"
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

        # Return evaluation metrics with class-wise accuracy
        return (
            float(loss),
            len(valloader.dataset),
            {"accuracy": float(accuracy), **class_accuracy},
        )


def main():
    # Parse command-line arguments
    args = parser.parse_args()
    print(args)

    # Ensure the client ID is within the allowed range
    assert args.cid < NUM_CLIENTS

    # Create checkpoint directory if it is specified in the arguments
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Prepare the dataset by partitioning it into training and validation sets
    trainsets, valsets, _ = prepare_dataset()

    # Start the Flower client, passing the appropriate dataset partition and other configurations
    fl.client.start_client(
        server_address=args.server_address,  # Server address for gRPC connection
        client=FlowerClient(
            trainset=trainsets[args.cid],  # Client-specific training dataset
            valset=valsets[args.cid],  # Client-specific validation dataset
            checkpoint_dir=args.checkpoint_dir,  # Directory for saving checkpoints
            client_id=args.cid,  # Client identifier
        ).to_client(),  # Convert FlowerClient to client format compatible with Flower framework
    )


if __name__ == "__main__":
    main()  # Entry point for script execution
