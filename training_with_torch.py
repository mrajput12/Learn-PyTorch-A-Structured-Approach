import torch

# Create the input and output tensors
X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=torch.float32)

# Initialize weight to a small random value instead of zero
w = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)  # Changed from 0.0 to 0.1


# Forward pass
def forward(x):
    return w * x


# Loss function
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


# Training parameters
learning_rate = 0.01  # Learning rate
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    y_pred = forward(X)  # Forward pass
    l = loss(Y, y_pred)  # Calculate loss
    l.backward()  # Backward pass (compute gradients)

    with torch.no_grad():
        w -= learning_rate * w.grad  # Update weight
    w.grad.zero_()  # Zero the gradients for the next iteration

    # Printing the results for each epoch
    print(f'epoch {epoch + 1} : w={w:.3f}, loss={l:.8f}')  # print to show every epoch

print(f' predicted after training : f(5) = {forward(5):.3f}')















