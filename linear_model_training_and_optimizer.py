import torch
import torch.nn as nn

# 1) Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Linear layer

    def forward(self, x):
        return self.linear(x)

# 0) Training samples
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

# 0) Create a test sample
X_test = torch.tensor([[5]], dtype=torch.float32)  # Keep the shape consistent for testing

# 1) Initialize the model
model = LinearRegressionModel(input_size=n_features)

# Display prediction before training
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # Forward pass: predict using the model
    y_predicted = model(X)

    # Calculate loss
    loss = loss_fn(y_predicted, Y)

    # Backward pass: compute gradients
    loss.backward()

    # Update weights
    optimizer.step()

    # Zero the gradients after updating
    optimizer.zero_grad()

    # Print the parameters and loss every 10 epochs
    if epoch % 10 == 0:
        weight, bias = model.parameters()  # Unpack parameters
        print(f'Epoch {epoch + 1}: Weight = {weight.item():.3f}, Bias = {bias.item():.3f}, Loss = {loss.item():.8f}')

# Display prediction after training
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
