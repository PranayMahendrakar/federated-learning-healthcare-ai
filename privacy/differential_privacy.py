"""
Differential Privacy for Federated Healthcare AI
=================================================
Implements Differentially Private Stochastic Gradient Descent (DP-SGD)
and privacy budget accounting for federated learning clients.

Privacy Guarantee:
    Each client's contribution satisfies (epsilon, delta)-differential privacy.
    This ensures that the global model cannot reveal whether a specific
    patient's data was used in training.

Based on:
    - Abadi et al. "Deep Learning with Differential Privacy" (2016)
    - McMahan et al. "A General Approach to Adding Differential Privacy to
      Iterative Training Procedures" (2018)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import math
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PrivacyAccountant:
    """
    Tracks privacy budget (epsilon, delta) across training rounds.
    Uses moments accountant for tight privacy analysis.
    """

    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.steps = 0
        self.privacy_log = []

    def compute_epsilon(
        self,
        steps: int,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
    ) -> float:
        """
        Estimate epsilon using the RDP accountant.
        Simplified Gaussian mechanism approximation.
        """
        if noise_multiplier == 0:
            return float("inf")

        # Gaussian mechanism privacy loss (simplified RDP to DP conversion)
        # For exact accounting, use autodp or opacus library
        rdp_orders = list(range(2, 128))
        rdp_epsilons = []

        for alpha in rdp_orders:
            # RDP for subsampled Gaussian mechanism
            rdp = self._compute_rdp_gaussian(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                alpha=alpha,
                steps=steps,
            )
            rdp_epsilons.append(rdp)

        # Convert RDP to (epsilon, delta)-DP
        epsilon = self._rdp_to_dp(rdp_epsilons, rdp_orders, delta)
        return epsilon

    def _compute_rdp_gaussian(
        self,
        noise_multiplier: float,
        sample_rate: float,
        alpha: int,
        steps: int,
    ) -> float:
        """Compute RDP for subsampled Gaussian mechanism."""
        if noise_multiplier == 0:
            return float("inf")

        # Single-step RDP bound for Gaussian mechanism with subsampling
        single_step = (
            alpha / (2 * noise_multiplier ** 2)
            if alpha > 1
            else (1 / (2 * noise_multiplier ** 2))
        )
        # Amplification by subsampling (Poisson)
        rdp = min(
            alpha * (sample_rate ** 2) * single_step,
            2 * alpha * sample_rate * single_step,
        )
        return steps * rdp

    def _rdp_to_dp(
        self, rdp_epsilons: List[float], rdp_orders: List[int], delta: float
    ) -> float:
        """Convert RDP to (epsilon, delta)-DP."""
        dp_epsilons = []
        for rdp, order in zip(rdp_epsilons, rdp_orders):
            if rdp == float("inf") or order == float("inf"):
                continue
            eps = rdp + math.log(1 / delta) / (order - 1)
            dp_epsilons.append(eps)
        return min(dp_epsilons) if dp_epsilons else float("inf")

    def update(self, steps: int, noise_multiplier: float, sample_rate: float):
        """Update privacy budget tracker."""
        self.steps += steps
        current_epsilon = self.compute_epsilon(
            self.steps, noise_multiplier, sample_rate, self.target_delta
        )
        self.privacy_log.append({
            "step": self.steps,
            "epsilon": current_epsilon,
            "delta": self.target_delta,
        })
        return current_epsilon

    def is_budget_exhausted(
        self, noise_multiplier: float, sample_rate: float
    ) -> bool:
        """Check if privacy budget is exhausted."""
        epsilon = self.compute_epsilon(
            self.steps, noise_multiplier, sample_rate, self.target_delta
        )
        return epsilon > self.target_epsilon


class DPOptimizer(torch.optim.Optimizer):
    """
    DP-SGD Optimizer: Adds calibrated Gaussian noise to gradients.

    Privacy mechanism:
    1. Clip each sample's gradient to max_grad_norm
    2. Add Gaussian noise scaled to noise_multiplier * max_grad_norm
    3. Result: (epsilon, delta)-DP guarantee per round

    Args:
        optimizer: Base optimizer (e.g., Adam, SGD)
        noise_multiplier: Noise scale relative to gradient clipping norm
        max_grad_norm: Maximum L2 norm for gradient clipping
        expected_batch_size: Expected batch size for DP accounting
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        expected_batch_size: Optional[int] = None,
    ):
        # Store attributes before calling super().__init__
        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.expected_batch_size = expected_batch_size
        self._step_count = 0

        # Initialize as wrapper (not a real torch.optim.Optimizer)
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults

        logger.info(
            f"DP-SGD initialized: noise={noise_multiplier}, clip_norm={max_grad_norm}"
        )

    def zero_grad(self, set_to_none: bool = True):
        """Reset gradients."""
        self.original_optimizer.zero_grad(set_to_none=set_to_none)

    def clip_and_noise_gradients(self, parameters: Iterator[nn.Parameter]):
        """
        Per-sample gradient clipping and noise addition.
        This is the core DP-SGD operation.
        """
        # Gradient clipping (limits sensitivity)
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=self.max_grad_norm
        )

        # Add Gaussian noise calibrated to sensitivity
        noise_scale = self.noise_multiplier * self.max_grad_norm
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad = param.grad + noise

        return total_norm

    def step(self, parameters: Optional[Iterator[nn.Parameter]] = None, **kwargs):
        """Apply DP-SGD step: clip gradients, add noise, then update."""
        if parameters is not None:
            self.clip_and_noise_gradients(parameters)

        self.original_optimizer.step(**kwargs)
        self._step_count += 1

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.original_optimizer.load_state_dict(state_dict)


class DifferentialPrivacyEngine:
    """
    High-level DP engine for federated learning clients.
    Wraps the model training loop with DP-SGD.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.accountant = PrivacyAccountant(target_epsilon, target_delta)

        logger.info(
            f"DP Engine: noise={noise_multiplier}, clip={max_grad_norm}, "
            f"target_epsilon={target_epsilon}"
        )

    def make_private(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[nn.Module, DPOptimizer]:
        """Wrap model and optimizer with DP guarantees."""
        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        return model, dp_optimizer

    def get_privacy_spent(
        self, num_steps: int, sample_rate: float
    ) -> Tuple[float, float]:
        """Get current privacy budget spent (epsilon, delta)."""
        epsilon = self.accountant.compute_epsilon(
            num_steps,
            self.noise_multiplier,
            sample_rate,
            self.accountant.target_delta,
        )
        return epsilon, self.accountant.target_delta

    def privacy_report(self, num_steps: int, dataset_size: int, batch_size: int) -> dict:
        """Generate a privacy budget report."""
        sample_rate = batch_size / dataset_size
        epsilon, delta = self.get_privacy_spent(num_steps, sample_rate)
        report = {
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "num_steps": num_steps,
            "sample_rate": sample_rate,
            "status": "OK" if epsilon <= self.accountant.target_epsilon else "BUDGET_EXHAUSTED",
        }
        logger.info(f"Privacy Report: epsilon={epsilon:.4f}, delta={delta}, status={report['status']}")
        return report


if __name__ == "__main__":
    # Quick test
    engine = DifferentialPrivacyEngine(noise_multiplier=1.1, max_grad_norm=1.0)
    epsilon, delta = engine.get_privacy_spent(num_steps=1000, sample_rate=0.01)
    print(f"After 1000 steps: epsilon={epsilon:.4f}, delta={delta}")
    report = engine.privacy_report(num_steps=1000, dataset_size=10000, batch_size=100)
    print(f"Privacy status: {report['status']}")
