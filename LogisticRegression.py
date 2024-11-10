import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 1) Define the model: A simple linear model with sigmoid activation
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation for output

# Instantiate the model
model = LogisticRegressionModel(n_features)

# 2) Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    y_pred = model(X_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Backward pass (gradient computation)
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients

    # Update weights
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4) Model evaluation
model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

with torch.no_grad():  # No gradient calculation during inference
    # Predict on the test set
    y_pred_test = model(X_test)
    y_pred_test_cls = y_pred_test.round()  # Convert probabilities to 0 or 1

    # Calculate accuracy
    correct = (y_pred_test_cls.eq(y_test)).sum().item()
    accuracy = correct / y_test.shape[0]
    print(f'Accuracy on test set: {accuracy:.4f}')
