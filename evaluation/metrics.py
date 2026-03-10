"""
Healthcare-Specific Evaluation Metrics for Federated AI
=========================================================
Comprehensive metrics for evaluating FL model performance
on medical time-series classification tasks.

Includes:
    - Standard classification metrics (accuracy, F1, AUC)
    - Healthcare-specific metrics (sensitivity, specificity, PPV, NPV)
    - Privacy-utility tradeoff analysis
    - Federated round metrics aggregation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import logging

logger = logging.getLogger(__name__)


# ─── Core Healthcare Metrics ──────────────────────────────────────── #

def compute_confusion_stats(
    y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1
) -> Dict[str, float]:
    """
    Compute confusion matrix statistics for binary classification.
    Essential for clinical decision-making evaluation.
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, pos_label]
    ).ravel()

    # Core clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # Negative Predictive Value
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    mcc = (
        (tp * tn - fp * fn)
        / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0
        else 0.0
    )

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "sensitivity": float(sensitivity),   # Recall / TPR
        "specificity": float(specificity),   # TNR
        "ppv": float(ppv),                   # Precision
        "npv": float(npv),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "mcc": float(mcc),                   # Matthews Correlation Coefficient
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Comprehensive classification metrics for ECG arrhythmia detection.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC computation)
        class_names: Names of classes for reporting
        average: Averaging strategy for multi-class metrics
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(per_class_f1):
        name = class_names[i] if class_names else f"class_{i}"
        metrics[f"f1_{name}"] = float(f1)

    # AUC-ROC (if probabilities provided)
    if y_prob is not None:
        n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
        if n_classes == 2:
            prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, prob_pos))
                metrics["avg_precision"] = float(average_precision_score(y_true, prob_pos))
            except Exception:
                metrics["auc_roc"] = 0.0
        else:
            try:
                metrics["auc_roc_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            except Exception:
                metrics["auc_roc_macro"] = 0.0

    return metrics


def compute_federated_metrics(
    round_num: int,
    client_metrics: List[Dict],
    global_metrics: Optional[Dict] = None,
    privacy_epsilon: Optional[float] = None,
) -> Dict:
    """
    Aggregate metrics across federated rounds and clients.
    
    Args:
        round_num: Current FL round number
        client_metrics: List of per-client metric dictionaries
        global_metrics: Global model evaluation metrics
        privacy_epsilon: Current privacy budget spent
    
    Returns:
        Aggregated metrics for this round
    """
    if not client_metrics:
        return {}

    # Aggregate client metrics (weighted average by dataset size)
    keys = [k for k in client_metrics[0].keys() if isinstance(client_metrics[0][k], (int, float))]
    aggregated = {}

    for key in keys:
        values = [m.get(key, 0) for m in client_metrics]
        aggregated[f"client_mean_{key}"] = float(np.mean(values))
        aggregated[f"client_std_{key}"] = float(np.std(values))
        aggregated[f"client_min_{key}"] = float(np.min(values))
        aggregated[f"client_max_{key}"] = float(np.max(values))

    result = {
        "round": round_num,
        "num_clients": len(client_metrics),
        **aggregated,
    }

    if global_metrics:
        result["global"] = global_metrics

    if privacy_epsilon is not None:
        result["privacy_epsilon"] = float(privacy_epsilon)

    return result


class MetricsTracker:
    """
    Track and log metrics across federated learning rounds.
    Useful for monitoring convergence and privacy-utility tradeoff.
    """

    def __init__(self, track_privacy: bool = True):
        self.round_history = []
        self.best_metrics = {}
        self.track_privacy = track_privacy

    def update(
        self,
        round_num: int,
        metrics: Dict,
        privacy_epsilon: Optional[float] = None,
    ):
        """Record metrics for a federated round."""
        entry = {
            "round": round_num,
            "metrics": metrics,
        }
        if privacy_epsilon is not None:
            entry["privacy_epsilon"] = privacy_epsilon

        self.round_history.append(entry)

        # Track best model
        acc = metrics.get("accuracy", metrics.get("global_accuracy", 0))
        if not self.best_metrics or acc > self.best_metrics.get("accuracy", 0):
            self.best_metrics = {"round": round_num, "accuracy": acc, **metrics}

        logger.info(
            f"Round {round_num}: accuracy={acc:.4f}"
            + (f", epsilon={privacy_epsilon:.4f}" if privacy_epsilon else "")
        )

    def get_summary(self) -> Dict:
        """Get training summary across all rounds."""
        if not self.round_history:
            return {}

        rounds = [e["round"] for e in self.round_history]
        accuracies = [
            e["metrics"].get("accuracy", 0) for e in self.round_history
        ]
        epsilons = [
            e.get("privacy_epsilon", None) for e in self.round_history
        ]

        return {
            "total_rounds": len(rounds),
            "best_accuracy": float(max(accuracies)),
            "best_round": rounds[accuracies.index(max(accuracies))],
            "final_accuracy": float(accuracies[-1]),
            "accuracy_improvement": float(accuracies[-1] - accuracies[0]),
            "final_epsilon": epsilons[-1] if any(e is not None for e in epsilons) else None,
            "best_metrics": self.best_metrics,
        }

    def print_report(self):
        """Print a formatted training report."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("FEDERATED LEARNING TRAINING REPORT")
        print("=" * 60)
        print(f"Total Rounds:      {summary.get('total_rounds', 0)}")
        print(f"Best Accuracy:     {summary.get('best_accuracy', 0):.4f} (Round {summary.get('best_round', 0)})")
        print(f"Final Accuracy:    {summary.get('final_accuracy', 0):.4f}")
        if summary.get('final_epsilon'):
            print(f"Privacy Spent:     epsilon = {summary['final_epsilon']:.4f}")
        print("=" * 60)


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: str = "cpu",
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a model on a dataset and return comprehensive metrics.
    Used for both local evaluation at hospitals and global evaluation.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_classification_metrics(
        all_labels, all_preds, all_probs, class_names
    )
    metrics["loss"] = total_loss / len(data_loader)
    metrics["num_samples"] = len(all_labels)

    return metrics
