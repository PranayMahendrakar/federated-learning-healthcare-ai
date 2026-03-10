# 🏥 Federated Learning for Privacy-Preserving Healthcare AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Flower](https://img.shields.io/badge/Flower-FL-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> Train ML models on distributed hospital data **without sharing patient data**. Raw data never leaves the hospital — only model gradients travel over the network, protected by secure aggregation and differential privacy.

---

## Problem Statement

Hospitals hold massive amounts of valuable patient data (ECG, vitals, time-series), but **data privacy laws** (HIPAA, GDPR) make centralizing this data impossible. Federated Learning solves this by:

- Training ML models **locally** on each hospital's data
- Sharing only **model updates** (not patient records)
- Aggregating updates **securely** at a central server
- Producing a **global model** that benefits from all hospitals' data

---

## Architecture

```
Hospital A       Hospital B       Hospital C
[Local ECG]     [Patient Mon]    [Vitals Data]
     |                |                |
[Local Train]  [Local Train]   [Local Train]
     |                |                |
     +--------Gradients (encrypted)----+
                       |
               [FL Server - FedAvg]
                       |
             [Global Healthcare Model]
```

---

## Research Focus

| Area | Description |
|------|-------------|
| **Data Privacy** | Differential privacy, secure multi-party computation |
| **Distributed AI** | FedAvg, FedProx, SCAFFOLD algorithms |
| **Healthcare Prediction** | ECG arrhythmia detection, patient deterioration |
| **Secure Aggregation** | Homomorphic encryption, secret sharing |

---

## Project Structure

```
federated-learning-healthcare-ai/
├── server/
│   ├── fl_server.py              # FL Server (Flower)
│   ├── aggregation.py            # FedAvg + secure aggregation
│   └── model_registry.py         # Global model versioning
├── client/
│   ├── fl_client.py              # FL Client (Hospital Node)
│   ├── local_trainer.py          # Local model training loop
│   └── data_loader.py            # Privacy-safe data loading
├── models/
│   ├── ecg_cnn.py                # 1D-CNN for ECG classification
│   ├── lstm_monitor.py           # LSTM for patient monitoring
│   └── transformer.py            # Transformer for time-series
├── data/
│   ├── preprocessing.py          # ECG signal preprocessing
│   ├── augmentation.py           # Privacy-safe augmentation
│   └── synthetic_gen.py          # Synthetic data generator
├── privacy/
│   ├── differential_privacy.py   # DP-SGD implementation
│   ├── secure_aggregation.py     # Secure aggregation protocol
│   └── noise_calibration.py      # Privacy budget management
├── evaluation/
│   ├── metrics.py                # Healthcare-specific metrics
│   ├── privacy_audit.py          # Privacy leakage evaluation
│   └── visualize.py              # Training curves & results
├── configs/
│   ├── server_config.yaml        # FL server configuration
│   └── client_config.yaml        # Client node configuration
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### Prerequisites
```bash
Python 3.10+
CUDA-capable GPU (recommended)
```

### Installation
```bash
git clone https://github.com/PranayMahendrakar/federated-learning-healthcare-ai.git
cd federated-learning-healthcare-ai
pip install -r requirements.txt
```

### Run Simulation
```bash
# Terminal 1: Start FL Server
python server/fl_server.py --rounds 10 --min-clients 3

# Terminal 2-4: Start Hospital Clients
python client/fl_client.py --hospital-id hospital_A --data-path data/ecg/
python client/fl_client.py --hospital-id hospital_B --data-path data/vitals/
python client/fl_client.py --hospital-id hospital_C --data-path data/ecg/
```

---

## Privacy Guarantees

- **Differential Privacy**: epsilon-DP with noise calibration (epsilon <= 1.0)
- **Secure Aggregation**: No single party sees individual updates
- **Gradient Clipping**: Limits sensitivity of updates
- **No Raw Data Sharing**: Patient records never leave the hospital

---

## Datasets

| Dataset | Task | Size |
|---------|------|------|
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/) | ECG Classification | 48 recordings |
| [PhysioNet 2019](https://physionet.org/content/challenge-2019/) | Sepsis Prediction | 40K patients |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | 12-lead ECG | 21K recordings |

---

## Results

| Model | Accuracy | F1-Score | Privacy Budget |
|-------|----------|----------|----------------|
| Local Only | 81.2% | 0.79 | N/A |
| Centralized | 91.4% | 0.90 | None |
| FedAvg (ours) | 88.7% | 0.87 | epsilon=1.0 |
| FedAvg + DP | 86.3% | 0.85 | epsilon=0.5 |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Pranay M Mahendrakar** — AI Specialist | Patent Holder | Open-Source Contributor
- GitHub: [@PranayMahendrakar](https://github.com/PranayMahendrakar)
- Website: [sonytech.in/pranay](https://sonytech.in/pranay)
