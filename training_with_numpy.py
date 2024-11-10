import numpy as np

# Input and output data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
Y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.float64)

# Initialize weight to a small random value instead of zero
w = 0.0

# Forward pass
def forward(x):
    return w * x

# Loss function
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

# Gradient calculation
def gradient(x, y, y_predicted):
    return np.dot(2 * x, (y_predicted - y)).mean()

# Training
learning_rate = 0.001  # Reduced learning rate
num_epochs = 3

for epoch in range(num_epochs):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    grad = gradient(X, Y, y_pred)
    w -= learning_rate * grad

    # Printing the results for each epoch
    print(f'epoch: {epoch + 1}, loss: {l:.4f}, w: {w:.4f}')


















