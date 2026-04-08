# MNIST Digit Classification — Progressive CUDA Optimization

A 2-layer MLP (Multi-Layer Perceptron) trained on MNIST, progressively optimized from high-level Python all the way down to custom CUDA kernels. The goal is to understand the performance characteristics at each level of abstraction and measure the speedup gained at each step.

---

## Architecture

A simple 2-layer fully connected network:

```
Input (784) → Linear(784→256) → ReLU → Linear(256→10) → Output (10)
```

- **Input:** 28×28 flattened MNIST images (784 features)
- **Hidden layer:** 256 neurons + ReLU activation
- **Output layer:** 10 neurons (one per digit class)
- **Loss:** Cross-Entropy

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Training samples used | 10,000 per epoch |
| Batch size | 4 |
| Epochs | 3 |
| Optimizer | SGD |
| Learning rate | 1e-3 |
| Normalization | mean=0.1307, std=0.3081 |

---

## Optimization Stages

| Stage | Implementation | Description |
|-------|---------------|-------------|
| 1 | Python / PyTorch | Baseline using high-level PyTorch API |
| 2 | NumPy | Manual forward/backward pass, no autograd |
| 3 | C / CPU | Pure C implementation, single-threaded |
| 4 | Naive CUDA | Custom CUDA kernels, no libraries |
| 5 | cuBLAS | CUDA with cuBLAS for matrix operations |
| ... | ... | Further optimizations (cuDNN, mixed precision, etc.) |

---

## Results

| Stage | Avg Iter Time | Time / Epoch | Speedup | Final Accuracy |
|-------|--------------|--------------|---------|----------------|
| PyTorch (CPU) | ~0.45 ms | ~1.1 s | 1× (baseline) | 90.51% |
| NumPy | — | — | — | — |
| C / CPU | — | ~1.87 s | 0.59× | 88.90% |
| Naive CUDA | — | — | — | — |
| cuBLAS | — | — | — | — |

> Iter time is averaged over logged checkpoints (every 500 iters). Epoch time estimated as avg\_iter\_time × iters\_per\_epoch.

---

## Project Structure

```
MNIST_Training_CUDA/
├── README.md
├── requirement.txt
├── 01_Python/
│   └── 01_torch_implementation.py   # Stage 1: PyTorch baseline
├── 02_NumPy/                        # Stage 2: NumPy (coming)
├── 02_C/                            # Stage 3: C/CPU implementation
│   ├── naive_cpu.c
│   └── export_data.py               # Exports MNIST to binary for C
├── 04_CUDA_Naive/                   # Stage 4: Naive CUDA kernels (coming)
├── 05_cuBLAS/                       # Stage 5: cuBLAS optimized (coming)
└── data/                            # MNIST dataset (auto-downloaded, gitignored)
```

---

## Requirements

```
torch
torchvision
numpy
matplotlib
```

Install with:

```bash
pip install -r requirement.txt
```

For CUDA stages, you will need:
- CUDA Toolkit (nvcc)
- A CUDA-capable GPU
- cuBLAS (included with CUDA Toolkit)
