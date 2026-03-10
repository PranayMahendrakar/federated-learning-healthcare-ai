"""
1D Convolutional Neural Network for ECG Arrhythmia Classification
==================================================================
Deep neural network for classifying cardiac arrhythmias from
single-lead or multi-lead ECG time-series signals.

Architecture: Multi-scale 1D CNN + Residual Blocks + Attention + Classifier
Designed for federated learning - lightweight yet accurate.

Classes (MIT-BIH):
    0: Normal beat (N)
    1: Supraventricular premature beat (S)
    2: Ventricular premature beat (V)
    3: Fusion of ventricular and normal beat (F)
    4: Unclassifiable beat (Q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock1D(nn.Module):
    """Residual block for 1D CNN — preserves gradient flow during FL."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for ECG — focuses on arrhythmia events."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(x)
        return x * attn_weights


class ECGClassifier(nn.Module):
    """
    Multi-scale 1D CNN with residual connections and attention
    for ECG arrhythmia classification in federated healthcare AI.

    Args:
        in_channels: Number of ECG leads (1 for single-lead, 12 for 12-lead)
        num_classes: Number of arrhythmia classes (default: 5 for MIT-BIH)
        sequence_length: Length of ECG segment in samples (default: 360)
        base_filters: Base number of convolutional filters
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        sequence_length: int = 360,
        base_filters: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Multi-scale feature extraction
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, base_filters // 4, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(base_filters // 4),
                nn.ReLU(),
            )
            for k in [3, 7, 15, 31]  # Multiple receptive fields
        ])

        # Deep feature extraction
        self.feature_extractor = nn.Sequential(
            # Stage 1
            nn.Conv1d(base_filters, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2),

            # Stage 2
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            ResidualBlock1D(base_filters * 2, dropout=dropout / 2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Stage 3
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
            ResidualBlock1D(base_filters * 4, dropout=dropout / 2),
            ResidualBlock1D(base_filters * 4, dropout=dropout / 2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Temporal attention
        self.attention = TemporalAttention(base_filters * 4)

        # Global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: ECG signal tensor of shape (batch, channels, sequence_length)
        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Multi-scale feature extraction
        ms_features = [branch(x) for branch in self.multi_scale]
        x = torch.cat(ms_features, dim=1)  # (batch, base_filters, length)

        # Deep feature extraction
        x = self.feature_extractor(x)

        # Apply temporal attention
        x = self.attention(x)

        # Global pooling + classify
        x = self.global_pool(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps before classification (for visualization)."""
        ms_features = [branch(x) for branch in self.multi_scale]
        x = torch.cat(ms_features, dim=1)
        x = self.feature_extractor(x)
        return self.attention(x)


def create_ecg_model(num_classes: int = 5, **kwargs) -> ECGClassifier:
    """Factory function to create ECG classifier."""
    model = ECGClassifier(num_classes=num_classes, **kwargs)
    print(f"ECGClassifier created: {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test the model
    model = create_ecg_model(num_classes=5)
    batch = torch.randn(8, 1, 360)  # (batch=8, leads=1, samples=360)
    output = model(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predictions:  {torch.softmax(output, dim=1).round(decimals=3)}")
