import argparse
import os
from collections import OrderedDict
from logging import INFO
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision
import cv2
from pathlib import Path
import time
from torchvision.models import mobilenet_v3_small

# Add command line arguments
parser = argparse.ArgumentParser(description="Federated Learning Model Inference")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    # default="saved_models/model_latest.pt",
    help="Path to the model file",
)
parser.add_argument(
    "--rtsp_url",
    type=str,
    required=True,
    help="RTSP URL for video stream",
)
parser.add_argument(
    "--mnist",
    action="store_true",
    help="Use MNIST model architecture",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.1,
    help="Minimum confidence threshold for predictions",
)

class Net(nn.Module):
    """MNIST Model Architecture"""
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

class ModelManager:
    def __init__(self, model_path, use_mnist=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.use_mnist = use_mnist
        self.model = None
        self.last_modified = None
        self.load_model()
   
    def load_model(self):
        """Load or reload the model if it has been updated"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
       
        current_modified = os.path.getmtime(self.model_path)
        if self.last_modified != current_modified:
            print(f"Loading model from {self.model_path}")
            if self.use_mnist:
                self.model = Net().to(self.device)
            else:
                self.model = mobilenet_v3_small(num_classes=10).to(self.device)
           
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.last_modified = current_modified
            print(f"Model loaded successfully (Round: {checkpoint.get('round', 'N/A')})")
   
    def get_model(self):
        """Get the current model, checking for updates"""
        self.load_model()
        return self.model

class InferenceProcessor:
    def __init__(self, use_mnist=False):
        self.use_mnist = use_mnist
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        # Set up preprocessing based on model type
        if use_mnist:
            self.transform = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
       
        self.labels = {
            0: "airplane" if not use_mnist else "0",
            1: "automobile" if not use_mnist else "1",
            2: "bird" if not use_mnist else "2",
            3: "cat" if not use_mnist else "3",
            4: "deer" if not use_mnist else "4",
            5: "dog" if not use_mnist else "5",
            6: "frog" if not use_mnist else "6",
            7: "horse" if not use_mnist else "7",
            8: "ship" if not use_mnist else "8",
            9: "truck" if not use_mnist else "9"
        }
   
    def preprocess_frame(self, frame):
        """Preprocess a frame for inference"""
        if self.use_mnist:
            # Convert to grayscale for MNIST
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (28, 28))
            frame = PIL.Image.fromarray(frame)
        else:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = PIL.Image.fromarray(frame)
       
        frame = self.transform(frame)
        frame = frame.unsqueeze(0).to(self.device)
        return frame
   
    def get_prediction(self, outputs, confidence_threshold):
        """Get prediction and confidence score"""
        probabilities = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        confidence = confidence.item()
        prediction = prediction.item()
       
        if confidence < confidence_threshold:
            return "Low confidence", 0.0
       
        label = self.labels.get(prediction, "Unknown")
        return label, confidence

def run_inference(rtsp_url, model_manager, processor, confidence_threshold):
    """Run inference on RTSP stream"""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")
   
    print(f"Starting inference on {rtsp_url}")
    fps_time = time.time()
    frame_count = 0
    fps = 0.0  # Initialize fps variable
   
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame, attempting to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url)
                continue
           
            # Get current model (checks for updates)
            model = model_manager.get_model()
           
            # Process frame
            processed_frame = processor.preprocess_frame(frame)
           
            # Run inference
            with torch.no_grad():
                outputs = model(processed_frame)
           
            # Get prediction
            label, confidence = processor.get_prediction(outputs, confidence_threshold)
           
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
           
            # Display results
            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0, 255, 0), 2, cv2.LINE_AA)
           
            cv2.imshow('Inference', frame)
           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
           
    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    args = parser.parse_args()
   
    try:
        # Initialize model manager and processor
        model_manager = ModelManager(args.model_path, args.mnist)
        processor = InferenceProcessor(args.mnist)
       
        # Run inference
        run_inference(
            args.rtsp_url,
            model_manager,
            processor,
            args.confidence_threshold
        )
   
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
   
    return 0

if __name__ == "__main__":
    exit(main())
