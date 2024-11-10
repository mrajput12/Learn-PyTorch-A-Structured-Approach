import torch
import torch.nn as nn
import torch.optim as optim


# Neural Network Class (with separate activation functions)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        # Define layers
        self.linear1 = nn.Linear(input_size, hidden_size)  # First linear layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.linear2 = nn.Linear(hidden_size, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output

    def forward(self, x):
        # Forward pass through the network
        out = self.linear1(x)  # Linear transformation
        out = self.relu(out)  # ReLU activation
        out = self.linear2(out)  # Another linear transformation
        out = self.sigmoid(out)  # Sigmoid activation for binary output
        return out


#  Creating the model
input_size = 3  # Number of input features
hidden_size = 2  # Number of neurons in hidden layer
model = NeuralNet(input_size=input_size, hidden_size=hidden_size)

# Creating sample data (1 sample with 3 features)
# Shape: (batch_size, input_size)
input_data = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
# Target value (1 for positive class, 0 for negative class)
target_data = torch.tensor([[1]], dtype=torch.float32)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: compute predicted y by passing x to the model
    outputs = model(input_data)

    # Compute the loss
    loss = criterion(outputs, target_data)

    # Backward pass: compute gradients
    optimizer.zero_grad()  # Zero the gradients before backward pass
    loss.backward()  # Compute gradients

    # Update weights using the optimizer
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Final output after training
with torch.no_grad():  # No gradient computation needed during inference
    final_output = model(input_data)
    print(f'Final Output (Predicted Probability): {final_output.item():.4f}')

# Decision rule: if output > 0.5, classify as class 1, else class 0
prediction = (final_output > 0.5).float()
print(f'Predicted Class: {prediction.item()}')

