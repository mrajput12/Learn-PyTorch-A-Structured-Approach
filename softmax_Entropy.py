import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def softmax(self, logits):
        """
        Applies the softmax function to the logits to get probabilities.
        Softmax normalizes the values to lie between 0 and 1 with sum equal to 1.
        """
        exp_values = np.exp(logits - np.max(logits))  # Stability trick: subtract max value
        probabilities = exp_values / np.sum(exp_values, axis=0)
        return probabilities

    def compute_loss(self, actual, predicted):
        """
        Calculates the cross-entropy loss between actual labels and predicted probabilities.
        """
        # Add epsilon to avoid log(0) which is undefined
        EPS = 1e-15
        predicted = np.clip(predicted, EPS, 1 - EPS)

        # Calculate cross-entropy loss
        loss = -np.sum(actual * np.log(predicted))
        return loss

    def forward(self, logits, actual):
        """
        Combines softmax and cross-entropy to get final loss.

        Parameters:
        - logits: Raw output scores (not probabilities) from the model
        - actual: One-hot encoded true labels
        """
        # Get probabilities from logits
        predicted = self.softmax(logits)

        # Compute the loss
        return self.compute_loss(actual, predicted)



logits = np.array([2.0, 1.0, 0.1])
actual = np.array([1, 0, 0])  # Class 0 is the true class

# Instantiate the class
cross_entropy = CrossEntropyLoss()

# Compute cross-entropy loss
loss = cross_entropy.forward(logits, actual)
print(f'Cross-Entropy Loss: {loss:.4f}')
