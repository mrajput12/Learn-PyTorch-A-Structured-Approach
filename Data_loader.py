import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Set batch size dynamically, you can adjust this based on your preference
batch_size = 8  # You can change this value as needed

# Example with built-in MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor()])  # Convert images to tensors

# Load MNIST dataset
train_dataset_mnist = torchvision.datasets.MNIST(root='./data',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

# DataLoader for MNIST
train_loader_mnist = DataLoader(dataset=train_dataset_mnist,
                                batch_size=batch_size,  # Set dynamic batch_size here
                                shuffle=True)

# Inspecting a random batch from MNIST
dataiter_mnist = iter(train_loader_mnist)
inputs_mnist, targets_mnist = next(dataiter_mnist)
print("MNIST Inputs shape:", inputs_mnist.shape)  # Should be [batch_size, 1, 28, 28]
print("MNIST Targets shape:", targets_mnist.shape)  # Should be [batch_size]

# To visualize MNIST images (optional, if you want to see sample images)
# Display the first image from the batch
plt.figure(figsize=(10, 10))

# Calculate grid dimensions
grid_size = math.ceil(math.sqrt(batch_size))  # This ensures a square grid layout

# Display images in a grid (for visualization purposes)
for i in range(batch_size):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(inputs_mnist[i].squeeze(), cmap='gray')  # Squeeze to remove unnecessary dimension
    plt.title(f'Label: {targets_mnist[i]}')
    plt.axis('off')  # Turn off axis for better visualization

plt.show()

# Dummy Training loop example
num_epochs = 2
total_samples = len(train_dataset_mnist)
n_iterations = math.ceil(total_samples / batch_size)  # Calculate number of iterations per epoch
print(f"Total samples: {total_samples}, Total iterations per epoch: {n_iterations}")

# Dummy training loop (without actual training for this demo)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader_mnist):
        # Simulating training process with print statements
        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_iterations}], '
                  f'Batch size: {inputs.shape[0]}, Labels batch size: {labels.shape[0]}')
