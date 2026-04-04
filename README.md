# MNIST Digit Classification — Progressive CUDA Optimization

A 2-layer MLP (Multi-Layer Perceptron) trained on MNIST, progressively optimized from high-level Python all the way down to custom CUDA kernels. The goal is to understand the performance characteristics at each level of abstraction and measure the speedup gained at each step.

---

## Architecture

A simple 2-layer fully connected network:

```
Input (784) → Linear → ReLU → Linear → Softmax → Output (10)
```

- **Input:** 28×28 flattened MNIST images (784 features)
- **Hidden layer:** 128 neurons + ReLU activation
- **Output layer:** 10 neurons (one per digit class)
- **Loss:** Cross-Entropy

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

| Stage | Time (s/epoch) | Speedup | Final Loss |
|-------|---------------|---------|------------|
| PyTorch | — | 1× (baseline) | — |
| NumPy | — | — | — |
| C / CPU | — | — | — |
| Naive CUDA | — | — | — |
| cuBLAS | — | — | — |

> Table will be updated as each stage is implemented.

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

---

## Project Structure

```
MNIST_Training_CUDA/
├── README.md
├── requirement.txt
├── pytorch/          # Stage 1: PyTorch baseline
├── numpy/            # Stage 2: NumPy implementation
├── c_cpu/            # Stage 3: C/CPU implementation
├── cuda_naive/       # Stage 4: Naive CUDA kernels
├── cuda_cublas/      # Stage 5: cuBLAS optimized
└── data/             # MNIST dataset (auto-downloaded)
```

---

## Training Details

- **Dataset:** MNIST (60,000 train / 10,000 test)
- **Optimizer:** SGD (or equivalent manual update)
- **Learning rate:** TBD per stage
- **Epochs:** TBD
- **Batch size:** TBD
