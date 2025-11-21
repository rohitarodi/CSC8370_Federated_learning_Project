# Quick Start Guide

Get up and running in 5 minutes!

## Setup (One Time)

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd CSC8370_Federated_learning_Project

# Or download and extract the ZIP file
```

### Step 2: Create Environment
```bash
# Using Conda (Recommended)
conda create -n datasec_env python=3.13
conda activate datasec_env

# OR using venv
python -m venv datasec_env
# Windows: datasec_env\Scripts\activate
# Linux/Mac: source datasec_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup
```bash
python test.py
```
If you see training progress and ~98% accuracy, you're ready!

---

## Run Solutions

### Level 1: Centralized Learning (~3 min)
```bash
python Templates/Level_1_Solution.py
```
Expected: 98-99% accuracy

### Level 2: Federated Learning (~7 min)
```bash
python Templates/Level_2_solution_1.py
```
Expected: 98-99% accuracy, 10 global rounds

### Level 3: Robust FL with Attack Detection (~8 min)
```bash
python Templates/Level_3_Solution.py
```
Expected: 98-99% accuracy, malicious client detected

---

## What Gets Created

After running all solutions, you'll have:
- `best_model.pth` - Level 1 model
- `federated_model.pth` - Level 2 model
- `robust_federated_model.pth` - Level 3 model
- `data/mnist/` - MNIST dataset (auto-downloaded)

---

## Need More Info?

See [README.md](README.md) for detailed documentation.

---

**That's it! You're ready to run the project.**
