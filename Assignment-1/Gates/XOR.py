import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Input Data (XOR Gate)
# -----------------------------
# Each row represents one training example:
# [x1, x2]
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected outputs for the XOR gate
y = np.array([0, 1, 1, 0])

# -----------------------------
# Initial Perceptron Parameters
# -----------------------------

# Initial weight for input x1
w1 = 1.2

# Initial weight for input x2
w2 = 0.6

# Threshold value used in the step activation function
threshold = 1

# Learning rate controls the size of weight updates
learning_rate = 0.5

# Number of times the entire dataset is used for training
epochs = 10

# Display initial weights before training starts
print("Initial Weights")
print("w1 =", w1)
print("w2 =", w2)

# -----------------------------
# Training Phase
# -----------------------------
# Repeat the training process for the specified number of epochs
for epoch in range(epochs):

    print(f"\n========== Epoch {epoch + 1} ==========")

    # Traverse every training example
    for i in range(len(X)):

        # Extract input values
        x1 = X[i][0]
        x2 = X[i][1]

        # Calculate net input
        net = x1 * w1 + x2 * w2

        # Apply Step Activation Function
        if net >= threshold:
            prediction = 1
        else:
            prediction = 0

        # Calculate prediction error
        error = y[i] - prediction

        # Display current training sample information
        print(f"\nInput: ({x1}, {x2})")
        print("Actual:", y[i])
        print("Predicted:", prediction)
        print("Error:", error)

        # Update weights only if prediction is incorrect
        if error != 0:

            # Update weight connected to x1
            w1 = w1 + learning_rate * error * x1

            # Update weight connected to x2
            w2 = w2 + learning_rate * error * x2

            print("Weights Updated!")

        else:
            print("No Update Needed.")

        # Display updated weights
        print(f"w1 = {w1:.2f}, w2 = {w2:.2f}")

# -----------------------------
# Display Final Weights
# -----------------------------
print("\nFinal Weights")
print(f"w1 = {w1:.2f}")
print(f"w2 = {w2:.2f}")

# -----------------------------
# Testing Phase
# -----------------------------
# Use the final trained weights to predict all inputs
print("\nTesting")

for i in range(len(X)):

    # Extract input values
    x1, x2 = X[i]

    # Calculate net input
    net = x1 * w1 + x2 * w2

    # Apply step activation function
    if net >= threshold:
        output = 1
    else:
        output = 0

    # Display prediction and expected output
    print(f"Input ({x1}, {x2}) -> Predicted: {output}, Actual: {y[i]}")

# -----------------------------
# Visualization
# -----------------------------

# Plot the input data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=150, edgecolors="black")

# Generate x-values for the decision boundary
x = np.array([-0.5, 1.5])

# Plot the decision boundary
if w2 != 0:
    y_line = (threshold - w1 * x) / w2
    plt.plot(x, y_line, "k-", label="Decision Boundary")

# Set graph limits
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# Add labels and title
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron XOR Gate")

# Display grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()