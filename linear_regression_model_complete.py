import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, num_epochs=100):
        # Initialize model parameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Define the linear model
        self.model = nn.Linear(1, 1)  # 1 input feature, 1 output
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        # Train the model
        for epoch in range(self.num_epochs):
            # Forward pass
            y_predicted = self.model(X)

            # Compute loss
            loss = self.criterion(y_predicted, y)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Zero gradients from previous step
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        # Make predictions
        with torch.no_grad():  # No need to track gradients
            return self.model(X).detach().numpy()

    def plot(self, X_numpy, y_numpy, predicted):
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(X_numpy, y_numpy, color='red', label='Original Data')  # Original data points
        plt.plot(X_numpy, predicted, color='blue', label='Fitted Line')  # Fitted line
        plt.title('Linear Regression with PyTorch')
        plt.xlabel('Input Feature')
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True)
        plt.show()


# Main execution
if __name__ == "__main__":
    # 0) Prepare data
    X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    # Convert numpy arrays to float Tensors
    X = torch.from_numpy(X_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)  # Reshape y to be a column vector

    # Initialize the model
    model = LinearRegressionModel(learning_rate=0.01, num_epochs=100)

    # Fit the model to the data
    model.fit(X, y)

    # Make predictions
    predicted = model.predict(X)

    # Plot the results
    model.plot(X_numpy, y_numpy, predicted)
