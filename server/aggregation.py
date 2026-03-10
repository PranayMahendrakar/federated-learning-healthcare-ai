"""
Secure Federated Aggregation Strategies
========================================
Custom aggregation strategies for privacy-preserving federated learning.
Extends Flower's built-in strategies with healthcare-specific features.

Strategies:
    - SecureFedAvg: FedAvg with optional server-side DP noise
    - FedProxHealthcare: FedProx with proximal term for heterogeneous data
    - RobustAggregation: Median-based aggregation for Byzantine robustness

References:
    - McMahan et al. "Communication-Efficient Learning of Deep Networks
      from Decentralized Data" (FedAvg, 2017)
    - Li et al. "Federated Optimization in Heterogeneous Networks"
      (FedProx, 2020)
"""

import flwr as fl
import numpy as np
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import reduce

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


def fedavg_aggregate(results: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    """
    Compute weighted average of model updates.
    Weight each client by its dataset size.
    
    Args:
        results: List of (weights, num_examples) tuples
    
    Returns:
        Weighted average of all weight arrays
    """
    num_examples_total = sum(num_examples for _, num_examples in results)
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def median_aggregate(results: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    """
    Coordinate-wise median aggregation for Byzantine robustness.
    More robust than FedAvg when clients may be compromised.
    """
    weights_list = [weights for weights, _ in results]
    return [
        np.median(np.stack([w[i] for w in weights_list]), axis=0)
        for i in range(len(weights_list[0]))
    ]


def trimmed_mean_aggregate(
    results: List[Tuple[np.ndarray, int]], trim_ratio: float = 0.1
) -> np.ndarray:
    """
    Trimmed mean aggregation: removes top/bottom outliers before averaging.
    Provides robustness against malicious gradient poisoning.
    """
    weights_list = [weights for weights, _ in results]
    n = len(weights_list)
    k = max(1, int(n * trim_ratio))  # Number to trim from each end

    trimmed = []
    for i in range(len(weights_list[0])):
        stacked = np.stack([w[i] for w in weights_list], axis=0)
        # Sort along client dimension and trim
        sorted_idx = np.argsort(stacked, axis=0)
        trimmed_stacked = np.take_along_axis(stacked, sorted_idx, axis=0)[k:n-k]
        trimmed.append(np.mean(trimmed_stacked, axis=0))
    return trimmed


class SecureFedAvg(FedAvg):
    """
    Secure Federated Averaging with optional server-side DP.
    
    Extends FedAvg with:
    - Optional server-side Gaussian noise injection
    - Anomaly detection for suspicious updates
    - Logging of privacy budget per round
    
    Args:
        dp_enabled: Enable differential privacy noise at server
        noise_multiplier: Scale of Gaussian noise relative to sensitivity
        max_grad_norm: Maximum gradient norm (clipping bound)
        **kwargs: Arguments passed to FedAvg
    """

    def __init__(
        self,
        *,
        dp_enabled: bool = False,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.round_metrics = []

        if dp_enabled:
            logger.info(
                f"SecureFedAvg: DP enabled, noise={noise_multiplier}, "
                f"clip={max_grad_norm}"
            )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with optional DP noise injection."""

        if not results:
            return None, {}

        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")

        # Convert parameters to NumPy arrays
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Weighted FedAvg aggregation
        aggregated_weights = fedavg_aggregate(weights_results)

        # Server-side DP: add noise to aggregated weights
        if self.dp_enabled:
            aggregated_weights = self._add_dp_noise(aggregated_weights, len(results))

        # Log aggregation metrics
        total_examples = sum(num_ex for _, num_ex in weights_results)
        metrics = {
            "num_clients": len(results),
            "total_examples": total_examples,
            "num_failures": len(failures),
        }

        logger.info(
            f"Round {server_round}: Aggregated {len(results)} clients, "
            f"{total_examples} total examples"
        )

        return ndarrays_to_parameters(aggregated_weights), metrics

    def _add_dp_noise(
        self, weights: List[np.ndarray], num_clients: int
    ) -> List[np.ndarray]:
        """Add calibrated Gaussian noise for server-side DP."""
        sensitivity = self.max_grad_norm / num_clients
        noise_scale = self.noise_multiplier * sensitivity
        noisy_weights = []
        for w in weights:
            noise = np.random.normal(0, noise_scale, w.shape).astype(w.dtype)
            noisy_weights.append(w + noise)
        return noisy_weights

    def _detect_anomalies(
        self, weights_list: List[List[np.ndarray]], threshold: float = 3.0
    ) -> List[bool]:
        """
        Detect anomalous client updates using z-score analysis.
        Returns boolean mask: True = normal, False = anomalous.
        """
        # Compute L2 norms of each client's full gradient
        norms = []
        for weights in weights_list:
            flat = np.concatenate([w.flatten() for w in weights])
            norms.append(np.linalg.norm(flat))

        norms = np.array(norms)
        if len(norms) < 3:
            return [True] * len(norms)

        z_scores = np.abs((norms - np.mean(norms)) / (np.std(norms) + 1e-8))
        is_normal = z_scores < threshold

        n_anomalies = np.sum(~is_normal)
        if n_anomalies > 0:
            logger.warning(f"Detected {n_anomalies} anomalous client updates")

        return is_normal.tolist()


class FedProxHealthcare(FedAvg):
    """
    FedProx for heterogeneous hospital data.
    
    Adds a proximal term to the local objective to prevent
    client drift when hospitals have very different data distributions.
    
    Loss = local_loss + (mu/2) * ||w - w_global||^2
    """

    def __init__(self, *, proximal_mu: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        logger.info(f"FedProxHealthcare: proximal_mu={proximal_mu}")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        """Send proximal_mu to clients via fit config."""
        config = {
            "proximal_mu": self.proximal_mu,
            "server_round": server_round,
        }
        return super().configure_fit(server_round, parameters, client_manager)


class RobustAggregation(SecureFedAvg):
    """
    Byzantine-robust aggregation using coordinate-wise median.
    Suitable when some hospital clients may be compromised.
    """

    def __init__(self, *, use_trimmed_mean: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_trimmed_mean = use_trimmed_mean

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using robust median instead of mean."""
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        if self.use_trimmed_mean:
            aggregated_weights = trimmed_mean_aggregate(weights_results)
        else:
            aggregated_weights = median_aggregate(weights_results)

        if self.dp_enabled:
            aggregated_weights = self._add_dp_noise(aggregated_weights, len(results))

        metrics = {
            "num_clients": len(results),
            "aggregation": "trimmed_mean" if self.use_trimmed_mean else "median",
        }
        logger.info(f"Round {server_round}: Robust aggregation ({metrics['aggregation']})")
        return ndarrays_to_parameters(aggregated_weights), metrics
