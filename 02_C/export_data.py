import numpy as np
from torchvision import datasets, transforms
import os

TRAIN_SIZE = 10000
TEST_SIZE = 1000
DATA_DIR = "./data"
OUT_DIR = "./data"

os.makedirs(OUT_DIR, exist_ok=True)

# Load raw pixel values (0.0 - 1.0), normalization is done in C
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=False)
test_dataset  = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=False)

# Export training data
X_train = np.array([train_dataset[i][0].numpy().flatten() for i in range(TRAIN_SIZE)], dtype=np.float32)
y_train = np.array([train_dataset[i][1] for i in range(TRAIN_SIZE)], dtype=np.int32)

# Export test data
X_test = np.array([test_dataset[i][0].numpy().flatten() for i in range(TEST_SIZE)], dtype=np.float32)
y_test = np.array([test_dataset[i][1] for i in range(TEST_SIZE)], dtype=np.int32)

X_train.tofile(f"{OUT_DIR}/X_train.bin")
y_train.tofile(f"{OUT_DIR}/y_train.bin")
X_test.tofile(f"{OUT_DIR}/X_test.bin")
y_test.tofile(f"{OUT_DIR}/y_test.bin")

print(f"Exported {TRAIN_SIZE} training samples and {TEST_SIZE} test samples")
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
