"""
ECG and Patient Monitoring Data Preprocessing
==============================================
Privacy-safe preprocessing pipeline for medical time-series data.
All operations are performed locally at each hospital — data never leaves.

Supports:
    - MIT-BIH Arrhythmia Database (ECG)
    - PhysioNet Challenge 2019 (Sepsis prediction)
    - PTB-XL (12-lead ECG)
    - Generic patient monitoring (ICU vitals)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.stats import zscore
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import logging
import wfdb

logger = logging.getLogger(__name__)


# ─── Signal Processing Utilities ───────────────────────────────────────────── #

def bandpass_filter(
    ecg_signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    fs: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to remove noise from ECG signal.
    
    Args:
        ecg_signal: Raw ECG signal array
        lowcut: Low cutoff frequency in Hz (removes baseline wander)
        highcut: High cutoff frequency in Hz (removes high-freq noise)
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered ECG signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, ecg_signal)


def remove_baseline_wander(
    ecg_signal: np.ndarray, fs: float = 360.0
) -> np.ndarray:
    """Remove baseline wander using median filter (privacy-preserving)."""
    window = int(0.2 * fs)
    if window % 2 == 0:
        window += 1
    baseline = signal.medfilt(ecg_signal, kernel_size=window)
    return ecg_signal - baseline


def normalize_signal(
    signal_data: np.ndarray, method: str = "zscore"
) -> np.ndarray:
    """
    Normalize ECG/vitals signal.
    
    Methods:
        zscore: Zero mean, unit variance
        minmax: Scale to [0, 1]
        robust: Robust scaling using IQR
    """
    if method == "zscore":
        std = np.std(signal_data)
        if std < 1e-8:
            return signal_data - np.mean(signal_data)
        return (signal_data - np.mean(signal_data)) / std
    elif method == "minmax":
        min_val, max_val = np.min(signal_data), np.max(signal_data)
        if max_val - min_val < 1e-8:
            return np.zeros_like(signal_data)
        return (signal_data - min_val) / (max_val - min_val)
    elif method == "robust":
        q25, q75 = np.percentile(signal_data, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-8:
            return signal_data - np.median(signal_data)
        return (signal_data - np.median(signal_data)) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def segment_ecg(
    ecg_signal: np.ndarray,
    segment_length: int = 360,
    overlap: float = 0.5,
    fs: float = 360.0,
) -> np.ndarray:
    """
    Segment long ECG recording into fixed-length windows.
    
    Args:
        ecg_signal: Full ECG recording
        segment_length: Samples per segment (default: 1 second at 360 Hz)
        overlap: Fraction of overlap between segments
        fs: Sampling frequency
    
    Returns:
        Array of segments: (num_segments, segment_length)
    """
    step = int(segment_length * (1 - overlap))
    segments = []
    for start in range(0, len(ecg_signal) - segment_length + 1, step):
        segment = ecg_signal[start:start + segment_length]
        segments.append(segment)
    return np.array(segments)


# ─── Datasets ─────────────────────────────────────────────────────────────── #

class MITBIHDataset(Dataset):
    """
    MIT-BIH Arrhythmia Database Dataset.
    
    Beat-level ECG classification dataset.
    Labels: N, S, V, F, Q (AAMI standard)
    """

    LABEL_MAP = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # Normal
        "A": 1, "a": 1, "J": 1, "S": 1,            # Supraventricular
        "V": 2, "E": 2,                              # Ventricular
        "F": 3,                                      # Fusion
        "/": 4, "f": 4, "Q": 4,                     # Unknown
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        segment_length: int = 360,
        transform=None,
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.segment_length = segment_length
        self.transform = transform

        self.segments = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """Load and preprocess MIT-BIH records."""
        records = list(self.data_dir.glob("*.dat"))
        if not records:
            logger.warning(f"No .dat files found in {self.data_dir}")
            self._generate_synthetic()
            return

        for record_path in records:
            record_name = str(record_path.with_suffix(""))
            try:
                record = wfdb.rdrecord(record_name, channels=[0])
                annotation = wfdb.rdann(record_name, "atr")
                ecg = record.p_signal[:, 0].astype(np.float32)

                # Preprocess
                ecg = bandpass_filter(ecg, fs=record.fs)
                ecg = remove_baseline_wander(ecg, fs=record.fs)

                # Extract beats around R-peaks
                for i, (sample, symbol) in enumerate(
                    zip(annotation.sample, annotation.symbol)
                ):
                    if symbol not in self.LABEL_MAP:
                        continue
                    half = self.segment_length // 2
                    start = max(0, sample - half)
                    end = start + self.segment_length
                    if end > len(ecg):
                        continue
                    beat = ecg[start:end]
                    beat = normalize_signal(beat, method="zscore")
                    self.segments.append(beat)
                    self.labels.append(self.LABEL_MAP[symbol])

            except Exception as e:
                logger.warning(f"Error loading {record_name}: {e}")

    def _generate_synthetic(self):
        """Generate synthetic ECG data for testing."""
        logger.info("Generating synthetic ECG data for testing")
        np.random.seed(42)
        for cls in range(5):
            for _ in range(200):
                t = np.linspace(0, 1, self.segment_length)
                freq = 1.2 + 0.1 * cls
                beat = (
                    np.sin(2 * np.pi * freq * t)
                    + 0.3 * np.sin(2 * np.pi * 3 * freq * t)
                    + 0.1 * np.random.randn(self.segment_length)
                )
                beat = normalize_signal(beat.astype(np.float32))
                self.segments.append(beat)
                self.labels.append(cls)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment = torch.tensor(self.segments[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            segment = self.transform(segment)
        return segment, label


class PatientMonitorDataset(Dataset):
    """
    ICU Patient Monitoring Dataset (PhysioNet 2019 format).
    For sepsis early prediction from vital signs time-series.
    """

    FEATURES = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
        "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
        "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct",
        "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_size: int = 12,
        prediction_horizon: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.samples = []
        self.targets = []
        self._load_data()

    def _load_data(self):
        """Load PSV files from PhysioNet 2019 format."""
        psv_files = list(self.data_dir.glob("*.psv"))
        if not psv_files:
            logger.warning("No PSV files found. Using synthetic data.")
            self._generate_synthetic()
            return

        for psv_file in psv_files[:1000]:  # Limit for demo
            try:
                df = pd.read_csv(psv_file, sep="|")
                features = df[self.FEATURES].values.astype(np.float32)
                labels = df["SepsisLabel"].values

                # Fill missing values with forward fill
                features = pd.DataFrame(features).ffill().bfill().fillna(0).values

                # Normalize each feature
                for i in range(features.shape[1]):
                    col = features[:, i]
                    std = np.std(col)
                    if std > 1e-8:
                        features[:, i] = (col - np.mean(col)) / std

                # Create sliding windows
                for t in range(len(features) - self.window_size - self.prediction_horizon + 1):
                    window = features[t:t + self.window_size]
                    label = int(labels[t + self.window_size + self.prediction_horizon - 1])
                    self.samples.append(window)
                    self.targets.append(label)

            except Exception as e:
                logger.debug(f"Error loading {psv_file.name}: {e}")

    def _generate_synthetic(self):
        """Generate synthetic patient monitoring data."""
        np.random.seed(42)
        n_samples = 2000
        for i in range(n_samples):
            label = np.random.randint(0, 2)
            noise_level = 0.3 if label == 1 else 0.1
            window = np.random.randn(self.window_size, len(self.FEATURES)).astype(np.float32)
            window += noise_level * np.random.randn(*window.shape)
            self.samples.append(window)
            self.targets.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.samples[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sample, target


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """Create DataLoader with privacy-safe settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Avoid revealing last batch size
    )
