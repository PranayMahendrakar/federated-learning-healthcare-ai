"""
Federated Learning Server for Privacy-Preserving Healthcare AI
=============================================================
Central server that orchestrates federated training across
multiple hospital nodes using the Flower (flwr) framework.

No patient data ever reaches this server - only encrypted
model updates (gradients) are aggregated here.
"""

import flwr as fl
import torch
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import numpy as np
from flwr.common import Metrics, Parameters, Scalar
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from aggregation import SecureFedAvg, FedProxHealthcare
from model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m.get("f1_score", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total = sum(examples)
    return {
        "accuracy": sum(accuracies) / total,
        "f1_score": sum(f1_scores) / total,
    }


def get_evaluate_fn(model, device: str = "cpu"):
    """Return evaluation function for server-side evaluation."""
    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Load global model weights
        params_dict = zip(model.state_dict().keys(), parameters.tensors)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # In production, evaluate on a held-out validation set
        # For simulation, return placeholder metrics
        logger.info(f"[Round {server_round}] Server-side evaluation complete")
        return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

    return evaluate


class FederatedHealthcareServer:
    """
    Main Federated Learning Server for Healthcare AI.
    
    Coordinates training across multiple hospital clients
    using privacy-preserving aggregation strategies.
    """

    def __init__(self, config_path: str = "configs/server_config.yaml"):
        self.config = self._load_config(config_path)
        self.registry = ModelRegistry()
        self.history = []

    def _load_config(self, config_path: str) -> dict:
        """Load server configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        return {
            "num_rounds": 10,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "server_address": "0.0.0.0:8080",
            "aggregation_strategy": "fedavg",
            "use_secure_aggregation": True,
            "differential_privacy": {
                "enabled": True,
                "noise_multiplier": 1.0,
                "max_grad_norm": 1.0,
            }
        }

    def build_strategy(self) -> fl.server.strategy.Strategy:
        """Build the federated learning aggregation strategy."""
        strategy_name = self.config.get("aggregation_strategy", "fedavg")
        dp_config = self.config.get("differential_privacy", {})

        if strategy_name == "fedprox":
            strategy = FedProxHealthcare(
                fraction_fit=self.config["fraction_fit"],
                fraction_evaluate=self.config["fraction_evaluate"],
                min_fit_clients=self.config["min_fit_clients"],
                min_evaluate_clients=self.config["min_evaluate_clients"],
                min_available_clients=self.config["min_available_clients"],
                evaluate_metrics_aggregation_fn=weighted_average,
                proximal_mu=0.1,
            )
        else:
            # Default: Secure FedAvg with optional DP
            strategy = SecureFedAvg(
                fraction_fit=self.config["fraction_fit"],
                fraction_evaluate=self.config["fraction_evaluate"],
                min_fit_clients=self.config["min_fit_clients"],
                min_evaluate_clients=self.config["min_evaluate_clients"],
                min_available_clients=self.config["min_available_clients"],
                evaluate_metrics_aggregation_fn=weighted_average,
                dp_enabled=dp_config.get("enabled", False),
                noise_multiplier=dp_config.get("noise_multiplier", 1.0),
                max_grad_norm=dp_config.get("max_grad_norm", 1.0),
            )

        logger.info(f"Using strategy: {strategy_name}")
        return strategy

    def start(self):
        """Start the federated learning server."""
        strategy = self.build_strategy()
        server_address = self.config.get("server_address", "0.0.0.0:8080")
        num_rounds = self.config.get("num_rounds", 10)

        logger.info(f"Starting FL server at {server_address}")
        logger.info(f"Training for {num_rounds} rounds")
        logger.info(f"Waiting for {self.config['min_available_clients']} clients...")

        history = fl.server.start_server(
            server_address=server_address,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        # Save final global model
        self.registry.save_model(history, round_num=num_rounds)
        logger.info("Training complete. Global model saved.")
        return history


def main():
    parser = argparse.ArgumentParser(description="FL Healthcare Server")
    parser.add_argument("--config", type=str, default="configs/server_config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--min-clients", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    server = FederatedHealthcareServer(config_path=args.config)

    # Override config with CLI args if provided
    if args.rounds:
        server.config["num_rounds"] = args.rounds
    if args.min_clients:
        server.config["min_fit_clients"] = args.min_clients
        server.config["min_evaluate_clients"] = args.min_clients
        server.config["min_available_clients"] = args.min_clients
    if args.host and args.port:
        server.config["server_address"] = f"{args.host}:{args.port}"

    history = server.start()

    # Print final results
    logger.info("=" * 50)
    logger.info("FEDERATED TRAINING COMPLETE")
    logger.info(f"Final accuracy: {history.metrics_distributed.get('accuracy', [])}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
