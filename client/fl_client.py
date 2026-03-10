"""
Federated Learning Client - Hospital Node
=========================================
Each hospital runs this client to participate in federated training.
Local patient data NEVER leaves the hospital. Only model updates
(gradients) are sent to the central server after encryption.
"""

import flwr as fl
import torch
import torch.nn as nn
import argparse
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from local_trainer import LocalTrainer
from data_loader import HealthcareDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareFlowerClient(fl.client.NumPyClient):
    """
    Flower client for a hospital node.

    Each hospital instantiates this client with its own local dataset.
    The client performs local training and returns model updates —
    never the raw patient data.
    """

    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        trainer: LocalTrainer,
        data_loader: HealthcareDataLoader,
        config: dict,
    ):
        self.hospital_id = hospital_id
        self.model = model
        self.trainer = trainer
        self.data_loader = data_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"[{hospital_id}] Client initialized on {self.device}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global model parameters into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Perform local training on hospital data.

        1. Load global model weights
        2. Train locally for N epochs
        3. Return updated weights (NOT patient data)
        """
        self.set_parameters(parameters)

        # Get training config from server
        local_epochs = int(config.get("local_epochs", 3))
        learning_rate = float(config.get("learning_rate", 1e-3))
        batch_size = int(config.get("batch_size", 32))

        logger.info(
            f"[{self.hospital_id}] Local training: {local_epochs} epochs, "
            f"lr={learning_rate}, batch_size={batch_size}"
        )

        # Train locally - patient data stays here
        train_loader = self.data_loader.get_train_loader(batch_size=batch_size)
        results = self.trainer.train(
            model=self.model,
            train_loader=train_loader,
            epochs=local_epochs,
            learning_rate=learning_rate,
        )

        logger.info(
            f"[{self.hospital_id}] Training complete. "
            f"Loss: {results['loss']:.4f}, Acc: {results['accuracy']:.4f}"
        )

        return (
            self.get_parameters(config={}),
            len(train_loader.dataset),
            {
                "loss": results["loss"],
                "accuracy": results["accuracy"],
                "hospital_id": self.hospital_id,
            },
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate global model on local validation data."""
        self.set_parameters(parameters)

        val_loader = self.data_loader.get_val_loader()
        loss, accuracy, f1 = self.trainer.evaluate(
            model=self.model,
            data_loader=val_loader,
        )

        logger.info(
            f"[{self.hospital_id}] Evaluation — "
            f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}"
        )

        return (
            float(loss),
            len(val_loader.dataset),
            {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "hospital_id": self.hospital_id,
            },
        )


def build_client(args, config: dict) -> HealthcareFlowerClient:
    """Build and configure the FL client."""
    from models.ecg_cnn import ECGClassifier
    from models.lstm_monitor import LSTMPatientMonitor

    # Select model based on data type
    data_type = config.get("data_type", "ecg")
    num_classes = config.get("num_classes", 5)

    if data_type == "ecg":
        model = ECGClassifier(
            in_channels=1,
            num_classes=num_classes,
            sequence_length=config.get("sequence_length", 360),
        )
    else:
        model = LSTMPatientMonitor(
            input_size=config.get("input_features", 34),
            hidden_size=128,
            num_classes=num_classes,
        )

    data_loader = HealthcareDataLoader(
        data_path=args.data_path,
        data_type=data_type,
        hospital_id=args.hospital_id,
        sequence_length=config.get("sequence_length", 360),
    )

    trainer = LocalTrainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_dp=config.get("differential_privacy", {}).get("enabled", False),
        dp_noise=config.get("differential_privacy", {}).get("noise_multiplier", 1.0),
        dp_clip=config.get("differential_privacy", {}).get("max_grad_norm", 1.0),
    )

    return HealthcareFlowerClient(
        hospital_id=args.hospital_id,
        model=model,
        trainer=trainer,
        data_loader=data_loader,
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(description="FL Healthcare Client (Hospital Node)")
    parser.add_argument("--hospital-id", type=str, required=True,
                        help="Unique identifier for this hospital")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to local patient data directory")
    parser.add_argument("--server-address", type=str, default="localhost:8080",
                        help="FL server address (host:port)")
    parser.add_argument("--config", type=str, default="configs/client_config.yaml")
    args = parser.parse_args()

    # Load client configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data_type": "ecg",
            "num_classes": 5,
            "sequence_length": 360,
            "differential_privacy": {"enabled": True, "noise_multiplier": 1.0},
        }

    logger.info(f"Hospital {args.hospital_id} connecting to server {args.server_address}")

    # Build and start the client
    client = build_client(args, config)

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
