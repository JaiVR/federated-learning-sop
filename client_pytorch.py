import argparse
import warnings
from collections import OrderedDict
import os
from pathlib import Path

import flwr as fl
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
    "--mnist",
    action="store_true",
    help="If you use Raspberry Pi Zero clients (which just have 512MB or RAM) use MNIST",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints",
    help="Directory to save client checkpoints",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(
    net, trainloader, optimizer, epochs, device, checkpoint_dir=None, client_id=None
):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = list(batch.values())
            images, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Save checkpoint after each epoch if directory is specified
        if checkpoint_dir is not None and client_id is not None:
            checkpoint_path = (
                Path(checkpoint_dir) / f"client_{client_id}_epoch_{epoch+1}.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss,
                },
                checkpoint_path,
            )

        print(f"Epoch {epoch+1} loss: {running_loss/len(trainloader):.3f}")


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    net.eval()

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            batch = list(batch.values())
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy


# Dataset preparation function remains the same
def prepare_dataset(use_mnist: bool):
    """Get MNIST/CIFAR-10 and return client partitions and global testset."""
    if use_mnist:
        fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    else:
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
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=42)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    return trainsets, validsets, testset


class FlowerClient(fl.client.NumPyClient):
    """Enhanced FlowerClient with improved model management and checkpointing."""

    def __init__(
        self, trainset, valset, use_mnist, checkpoint_dir=None, client_id=None
    ):
        self.trainset = trainset
        self.valset = valset
        self.checkpoint_dir = checkpoint_dir
        self.client_id = client_id

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Instantiate model
        if use_mnist:
            self.model = Net()
        else:
            self.model = mobilenet_v3_small(num_classes=10)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

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

        batch_size = config["batch_size"]
        epochs = config["epochs"]

        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Important for embedded devices
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

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation dataset."""
        print(f"Client {self.client_id}: Starting evaluate()")
        self.set_parameters(parameters)

        valloader = DataLoader(
            self.valset, batch_size=64, num_workers=0  # Important for embedded devices
        )

        loss, accuracy = test(self.model, valloader, self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # Create checkpoint directory if specified
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Download dataset and partition it
    trainsets, valsets, _ = prepare_dataset(args.mnist)

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid],
            valset=valsets[args.cid],
            use_mnist=args.mnist,
            checkpoint_dir=args.checkpoint_dir,
            client_id=args.cid,
        ).to_client(),
    )


if __name__ == "__main__":
    main()
