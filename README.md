# CSC8370_25FullDemo - MNIST Federated Learning with Byzantine Attack Detection

A comprehensive implementation of federated learning for MNIST digit classification, featuring robust Byzantine attack detection and mitigation.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Running the Solutions](#running-the-solutions)
- [Implementation Details](#implementation-details)
- [Results Summary](#results-summary)

---

## Project Overview

This project implements three levels of machine learning systems for MNIST classification:

### **Level 1: Centralized Learning**
- Traditional centralized training approach
- Single model trained on entire dataset
- Baseline for comparison

### **Level 2: Federated Learning**
- Distributed training across 10 clients
- Privacy-preserving collaborative learning
- FedAvg aggregation algorithm

### **Level 3: Robust Federated Learning**
- Byzantine attack simulation (Model Poisoning)
- Pairwise cosine similarity detection
- Malicious client exclusion mechanism
- Maintains accuracy despite attacks

---

## Requirements

### **System Requirements**
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)

### **Python Package Requirements**
All required packages are listed in `requirements.txt`:
- PyTorch 2.9.0+
- TorchVision 0.24.0+
- NumPy 2.2.4+
- Matplotlib 3.10.1+ (optional, for visualization)

---

## Environment Setup

### **Option 1: Using Conda (Recommended)**

1. **Create a new conda environment:**
   ```bash
   conda create -n datasec_env python=3.13
   ```

2. **Activate the environment:**
   ```bash
   conda activate datasec_env
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### **Option 2: Using venv (Alternative)**

1. **Create a virtual environment:**
   ```bash
   python -m venv datasec_env
   ```

2. **Activate the environment:**
   - **Windows:**
     ```bash
     datasec_env\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source datasec_env/bin/activate
     ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### **Option 3: Install Packages Directly**

If you encounter issues with the requirements file, install packages individually:

```bash
pip install torch torchvision numpy matplotlib
```

### **Verify Installation**

Run the test script to verify your environment:
```bash
python test.py
```

**Expected output:** Training for 3 epochs, final accuracy ~98%

---

## Project Structure

```
CSC8370_Federated_learning_Project/
├── README.md                          # This file
├── requirements.txt           # Essential packages only
│
├── test.py                            # Environment verification script
├── best_model.pth                     # Saved model from test.py
│
├── Templates/                         # Solution implementations
│   ├── Level_1_Solution.py           # Centralized Learning
│   ├── Level_2_solution_1.py         # Federated Learning
│   ├── Level_3_Solution.py           # Robust FL with Attack Detection
│   ├── dataloader4level1.py          # (Reference)
│   └── a federated learning framework.py  # (Documentation)
│
└── data/                              # Auto-downloaded MNIST dataset
    └── mnist/
```

---

## Running the Solutions

### **Prerequisites**
Ensure your environment is activated:
```bash
conda activate datasec_env  # or your environment name
```

---

### **Level 1: Centralized Learning**

**Objective:** Train a CNN on the full MNIST dataset in a centralized manner.

**Command:**
```bash
python Templates/Level_1_Solution.py
```

**What it does:**
- Loads entire MNIST dataset (60,000 training images)
- Trains CNN for 3 epochs
- Evaluates on 10,000 test images
- Saves best model as `best_model.pth`

**Expected Output:**
```
The number of training data: 60000
The number of testing data: 10000
training has:1200 batch of data!
testing has:1 batch of data!
epoch:1,index of train:100,loss: 1.539659,acc:55.30%
...
epoch:3,index of train:1200,loss: 0.047619,acc:98.48%
Best model saved with accuracy: 0.9849
Accuracy: 0.9849
```

**Runtime:** ~2-5 minutes (CPU), ~1-2 minutes (GPU)

**Expected Accuracy:** 97-99%

---

### **Level 2: Federated Learning**

**Objective:** Implement federated learning with 10 clients.

**Command:**
```bash
python Templates/Level_2_solution_1.py
```

**What it does:**
- Splits MNIST dataset among 10 clients (6,000 images each)
- Each client trains locally for 2 epochs
- Server aggregates models using FedAvg
- Repeats for 10 global rounds
- Saves final model as `federated_model.pth`

**Expected Output:**
```
Global Epoch 1/10
Global Model Test Accuracy after round 1: 0.9039
Global Epoch 2/10
Global Model Test Accuracy after round 2: 0.9470
...
Global Epoch 10/10
Global Model Test Accuracy after round 10: 0.9851
Federated learning process completed.
```

**Runtime:** ~5-10 minutes (CPU), ~2-4 minutes (GPU)

**Expected Accuracy:** 97-99%

**Key Parameters (configurable at line 109):**
```python
federated_learning(
    n_clients=10,        # Number of clients
    global_epochs=10,    # Global rounds
    local_epochs=2       # Local training epochs per round
)
```

---

### **Level 3: Robust Federated Learning with Attack Detection**

**Objective:** Detect and mitigate Byzantine attacks in federated learning.

**Command:**
```bash
python Templates/Level_3_Solution.py
```

**What it does:**
- Simulates federated learning with 10 clients
- Client 5 becomes malicious at Round 5
- Malicious client injects random noise (model poisoning)
- Detection mechanism uses pairwise cosine similarity
- Excludes detected malicious clients from aggregation
- Maintains model accuracy despite ongoing attack
- Saves final model as `robust_federated_model.pth`

**Expected Output:**
```
======================================================================
ROBUST FEDERATED LEARNING WITH ATTACK DETECTION
======================================================================
Configuration:
  - Clients: 10
  - Global Epochs: 10
  - Local Epochs: 2
  - Malicious Client: 5
  - Attack Starts: Round 5
======================================================================

Global Epoch 1/10
  Running malicious client detection...
  Aggregating 10 benign clients (excluded 0 malicious)
  Global Model Test Accuracy: 0.8988
======================================================================
...
======================================================================
Global Epoch 5/10
  Client 5: MALICIOUS (injecting false updates)
  Running malicious client detection...
  Client 5 flagged as malicious (avg similarity: 0.0095)
  Aggregating 9 benign clients (excluded 1 malicious)
  Global Model Test Accuracy: 0.9753
  Malicious client 5 successfully detected!
======================================================================
...
ROBUST FEDERATED LEARNING COMPLETED
   Final Accuracy: 0.9851
======================================================================
```

**Runtime:** ~5-10 minutes (CPU), ~2-4 minutes (GPU)

**Expected Accuracy:** 97-99% (maintained despite attack!)

**Key Parameters (configurable at lines 244-248):**
```python
robust_federated_learning(
    n_clients=10,              # Number of clients
    global_epochs=10,          # Global rounds (required: 10)
    local_epochs=2,            # Local training epochs
    malicious_client_id=5,     # Which client is malicious
    attack_start_round=5       # When attack begins
)
```

---

## Implementation Details

### **CNN Architecture (ConvNet)**
```python
Input: 28x28 grayscale image
Conv1: 1→6 channels, 5×5 kernel → ReLU → MaxPool(2×2)
Conv2: 6→16 channels, 5×5 kernel → ReLU → MaxPool(2×2)
Flatten: 16×4×4 = 256
FC1: 256→120 → ReLU
FC2: 120→84 → ReLU
FC3: 84→10 (output classes)
```

### **Federated Learning (FedAvg)**
```python
For each global round:
    1. Distribute global model to all clients
    2. Each client trains locally on their data
    3. Clients send updated models to server
    4. Server averages all client models:
       global_params = mean(client_params[0], ..., client_params[9])
    5. Repeat
```

### **Attack Detection Mechanism**
```python
Pairwise Cosine Similarity Detection:
    1. For each client, compute similarity with all other clients
    2. Calculate average similarity for each client
    3. Benign clients: avg_similarity > 0.95 (very similar)
    4. Malicious client: avg_similarity < 0.1 (very different)
    5. Threshold: 0.8 (flag clients below this)
    6. Exclude flagged clients from aggregation
```

### **Attack Type**
- **Category:** Model Poisoning / Byzantine Attack
- **Method:** Random Gaussian noise injection (magnitude: 10.0)
- **Target:** Model parameters directly
- **Impact:** Without detection, accuracy drops to ~10-20%
- **With detection:** Accuracy maintained at ~98%

---

## Results Summary

| Metric | Level 1 | Level 2 | Level 3 |
|--------|---------|---------|---------|
| **Approach** | Centralized | Federated (10 clients) | Robust FL |
| **Training Mode** | Single server | Distributed | Distributed + Defense |
| **Epochs** | 3 | 10 global × 2 local | 10 global × 2 local |
| **Attack Present** | No | No | Yes (Round 5+) |
| **Detection Active** | N/A | N/A | Yes |
| **Final Accuracy** | 98.49% | 98.51% | 98.51% |
| **Detection Rate** | N/A | N/A | 100% (6/6 rounds) |
| **Runtime (CPU)** | ~3 min | ~7 min | ~8 min |

### **Key Findings:**
1. **Federated learning** achieves comparable accuracy to centralized learning
2. **Byzantine attack** successfully detected in all attack rounds
3. **Model accuracy maintained** at 98.51% despite ongoing attack
4. **Zero false positives** - only malicious client flagged
5. **System remains functional** with 9/10 clients after exclusion

---

## Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'torch'"**
**Solution:** Install PyTorch:
```bash
pip install torch torchvision
```

### **Issue: "CUDA out of memory"**
**Solution:** The code automatically falls back to CPU. If you want to force CPU:
```python
device = torch.device('cpu')  # Line 58/69/167 in solution files
```

### **Issue: "RuntimeError: DataLoader worker is killed"**
**Solution:** Reduce batch size or number of workers:
```python
batch_size = 25  # Reduce from 50
```

### **Issue: Dataset download fails**
**Solution:** Download MNIST manually from http://yann.lecun.com/exdb/mnist/ and place in `./data/mnist/`

### **Issue: Slow training on CPU**
**Solution:** Normal behavior. CPU training takes 2-3× longer than GPU. Be patient!

---

## Configuration Options

### **Hyperparameters (can be modified):**

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `learning_rate` | Level 1: line 62 | 0.0005 | Optimizer learning rate |
| `batch_size` | Level 1: line 14 | 50 | Training batch size |
| `epoches` | Level 1: line 63 | 3 | Training epochs |
| `n_clients` | Level 2: line 109 | 10 | Number of federated clients |
| `global_epochs` | Level 2: line 109 | 10 | Federated rounds |
| `local_epochs` | Level 2: line 109 | 2 | Client training epochs |
| `threshold` | Level 3: line 220 | 0.8 | Detection threshold |
| `malicious_client_id` | Level 3: line 247 | 5 | Which client attacks |
| `attack_start_round` | Level 3: line 248 | 5 | When attack begins |

---

## Additional Resources

### **Federated Learning:**
- Original FedAvg Paper: McMahan et al., 2017
- https://arxiv.org/abs/1602.05629

### **Byzantine Attacks:**
- Survey: Lyu et al., 2020
- https://arxiv.org/abs/2007.10747

### **Defense Mechanisms:**
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- https://arxiv.org/abs/1703.02757

---

## Authors & Acknowledgments

**Course:** CSC8370 - Data Security
**Institution:** Georgia State University
**Semester:** 2025

**Project Template:** Provided by TA Dong Yang
**Implementation:** Rohit Arodi Ramachandra (002830329) & Ashish Reddy Mandadi (002850578)

---

## Support

Email : rarodiramachandra1@student.gsu.edu

---

**Last Updated:** November 2025
**Version:** 1.0
