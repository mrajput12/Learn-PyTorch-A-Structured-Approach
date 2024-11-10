import torch
import torch.nn as nn


# 1) Design Model using PyTorch's nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # We are learning only 1 parameter, weight
        self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        return self.weight * x


# 0) Training samples
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Initialize model
model = LinearRegressionModel()

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

# Mean Squared Error loss
loss_fn = nn.MSELoss()

# Using Stochastic Gradient Descent (SGD) optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initial prediction before training
print(f'Prediction before training: f(5) = {model.forward(torch.tensor(5.0)).item():.3f}')

# 3) Training loop
for epoch in range(n_iters):
    # Forward pass: compute prediction
    y_predicted = model.forward(X)

    # Compute loss
    loss = loss_fn(y_predicted, Y)

    # Backward pass: compute gradients
    loss.backward()

    # Update weights
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad()

    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}: w = {model.weight.item():.3f}, loss = {loss.item():.8f}')

# Prediction after training
print(f'Prediction after training: f(5) = {model.forward(torch.tensor(5.0)).item():.3f}')








