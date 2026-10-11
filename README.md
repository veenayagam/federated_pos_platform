# Privacy-Preserving Federated AI for POS Networks

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated_Learning-green?logo=linux&logoColor=white)](https://flower.ai/)
[![Opacus](https://img.shields.io/badge/Opacus-Differential_Privacy-purple)](https://opacus.ai/)
[![Polars](https://img.shields.io/badge/Polars-Fast_Data-orange)](https://pola.rs/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An open-source, edge-computable Federated Learning pipeline designed to train cross-domain recommendation engines for independent U.S. Small-to-Medium Enterprises (SMEs). This system works without ever centralizing proprietary transaction records or consumer PII.

---

## Overview

Independent retail and hospitality businesses often face an algorithmic disadvantage compared to massive enterprise delivery platforms that pool consumer data across thousands of stores. Traditional AI models require merchants to upload their raw transaction logs into centralized third-party clouds. This not only exposes sensitive customer data but also triggers regulatory liability under frameworks like CCPA and CPRA.

This project solves that problem by implementing a Zero-Trust Federated AI Architecture. Rather than centralizing sales data, it keeps 100% of order logs strictly on the local merchant's hardware. The mathematical models train locally on Clover POS data exports, apply formal differential privacy noise to protect the data, and return only encrypted numerical weight updates to build a shared intelligence layer.

---

## System Architecture & Core Technologies

- Zero-Trust Ingestion (Polars): Automatically cleans raw Clover POS CSV exports. It discards Personally Identifiable Information (PII) like customer names and payment tokens, isolating only the item-level co-purchase pairs.
- Dual-Embedding Neural Network (PyTorch): Implements a dual-embedding classification network optimized for real-time edge execution, achieving sub-5ms local CPU inference latency.
- Cryptographic Privacy Engine (Opacus): Enforces local gradient clipping (max_grad_norm=1.0) and injects calibrated Gaussian noise (noise_multiplier=1.15) into the parameter updates.
- Federated Aggregation (Flower): Coordinates multi-node communication rounds using the FedAvg strategy, carefully tracking cumulative privacy expenditure across all decentralized iterations.

---

## Benchmark Results

We validated the system across a 3-node non-IID restaurant coalition, modeling distinct North Indian, South Indian, and Biryani/Grill menus. The benchmarks demonstrated:

- Recommendation Accuracy: ~53% (Cross-domain signal), successfully overcoming the severe overfitting (~35%) seen in isolated in-house models.
- Privacy Budget: Strictly bounded at epsilon < 3.0 (delta = 10^-4).
- Edge Inference Latency: < 5.00 ms on a local CPU, well under the 10 ms real-time POS target threshold.

---

## Step-by-Step Reproduction Guide

### Prerequisites

- Python 3.9 or higher
- Git

---

### Step 1: Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

Install all required distributed learning, differential privacy, and data-processing packages:

```bash
pip install -q "flwr[simulation]" polars opacus matplotlib torch streamlit
```

### Step 3: Run the 10-Round Federated Simulation

Execute the full multi-node benchmark modeling the 3-restaurant coalition:

```bash
python app.py
```

#### Execution Pipeline

- Synthetic Data Synthesis: Generates non-IID Clover POS transaction logs for the three simulated restaurants.
- Local Zero-Trust Ingestion: Ingests each CSV locally with Polars, strips PII, and builds training pairs.
- Baseline 1: Trains isolated PyTorch models at each node without federation for comparison.
- Benchmark 2: Spawns a 3-client Flower federated coalition with Opacus differential privacy for 10 communication rounds.
- Output Generation: Prints the comparative accuracy and privacy expenditure table in the terminal and saves a high-resolution benchmark visualization.

### Step 4: Run the Interactive POS Edge Terminal (Demo UI)

To launch the simulated cashier interface and test real-time CPU recommendation inference:

```bash
streamlit run app.py
```

Open <http://localhost:8501> to test the live upselling recommendations and verify the sub-5ms latency.

---

## Broader Impact & National Utility

This architecture bridges the technological divide between Big Tech conglomerates and independent Main Street businesses. By democratizing differential privacy and federated learning for standard POS environments, this repository provides a reproducible, open-source blueprint for decentralized machine learning.

Contributions from open-source developers, privacy researchers, and POS system integrators are always welcome.

---

## License

This project is licensed under the Apache 2.0 License.
