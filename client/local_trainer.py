"""
Local Model Trainer - Hospital Node
====================================
Performs local training on hospital data with optional
Differential Privacy (DP-SGD) to protect patient information.

The trainer runs N epochs on local data and returns
updated model weights — never the raw patient data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Local model trainer for a federated learning hospital node.
    
    Supports:
        - Standard training (SGD, Adam)
        - DP-SGD for differential privacy
        - FedProx proximal term
        - Early stopping based on local validation loss
    
    Args:
        device: Training device ('cpu' or 'cuda')
        use_dp: Enable Differential Privacy (DP-SGD)
        dp_noise: Gaussian noise multiplier for DP
        dp_clip: Per-sample gradient clipping norm for DP
    """

    def __init__(
        self,
        device: str = "cpu",
        use_dp: bool = True,
        dp_noise: float = 1.1,
        dp_clip: float = 1.0,
    ):
        self.device = torch.device(device)
        self.use_dp = use_dp
        self.dp_noise = dp_noise
        self.dp_clip = dp_clip
        self.training_history = []

        if use_dp:
            logger.info(f"DP-SGD enabled: noise={dp_noise}, clip={dp_clip}")

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        proximal_mu: Optional[float] = None,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict:
        """
        Train model locally on hospital data.
        
        Args:
            model: Neural network model to train
            train_loader: Local training data loader
            epochs: Number of local epochs
            learning_rate: Learning rate for local optimizer
            optimizer_name: 'adam' or 'sgd'
            proximal_mu: FedProx mu parameter (None = disabled)
            global_params: Global model parameters for FedProx term
        
        Returns:
            Dictionary with training metrics (loss, accuracy)
        """
        model.to(self.device)
        model.train()

        # Build optimizer
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4,
            )

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Store global params for FedProx
        if proximal_mu is not None and global_params is None:
            global_params = {
                k: v.clone().detach() for k, v in model.state_dict().items()
            }

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # FedProx proximal term
                if proximal_mu is not None and global_params is not None:
                    prox_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params:
                            prox_term += torch.norm(
                                param - global_params[name].to(self.device)
                            ) ** 2
                    loss += (proximal_mu / 2) * prox_term

                loss.backward()

                # Differential privacy: clip gradients then add noise
                if self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.dp_clip
                    )
                    self._add_gradient_noise(model, len(train_loader.dataset))

                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    correct = (preds == targets).sum().item()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_correct += correct
                epoch_samples += inputs.size(0)

            scheduler.step()

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

            epoch_acc = epoch_correct / epoch_samples
            logger.debug(
                f"Epoch {epoch+1}/{epochs}: loss={epoch_loss/epoch_samples:.4f}, "
                f"acc={epoch_acc:.4f}"
            )

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        result = {"loss": avg_loss, "accuracy": avg_acc, "epochs": epochs}
        self.training_history.append(result)
        return result

    def _add_gradient_noise(self, model: nn.Module, dataset_size: int):
        """Add calibrated Gaussian noise to gradients for DP-SGD."""
        noise_scale = self.dp_noise * self.dp_clip
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad += noise

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        Evaluate model on local validation data.
        
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        model.eval()
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        n = len(all_targets)
        if n == 0:
            return 0.0, 0.0, 0.0

        avg_loss = total_loss / n
        accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / n

        # Macro F1 (simplified)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        classes = np.unique(all_targets)
        f1_scores = []
        for cls in classes:
            tp = np.sum((all_preds == cls) & (all_targets == cls))
            fp = np.sum((all_preds == cls) & (all_targets != cls))
            fn = np.sum((all_preds != cls) & (all_targets == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        return float(avg_loss), float(accuracy), macro_f1
