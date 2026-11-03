import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("Student.csv")

# Display first few rows
display(df.head())

# -----------------------------
# Input Data
# -----------------------------

# Select two input features
X = df[["attendance_pct", "study_hours_per_week"]].values

# Target column
y = df["pass"].values

# -----------------------------
# Initial Perceptron Parameters
# -----------------------------

# Initial weights
w1 = 0.5
w2 = 0.5

# Threshold
threshold = 1

# Learning rate
learning_rate = 0.01

# Number of epochs
epochs = 10

print("Initial Weights")
print("w1 =", w1)
print("w2 =", w2)

# -----------------------------
# Training Phase
# -----------------------------

for epoch in range(epochs):

    print(f"\n========== Epoch {epoch + 1} ==========")

    for i in range(len(X)):

        # Extract input values
        x1 = X[i][0]
        x2 = X[i][1]

        # Calculate net input
        net = x1 * w1 + x2 * w2

        # Step activation function
        if net >= threshold:
            prediction = 1
        else:
            prediction = 0

        # Calculate error
        error = y[i] - prediction

        # Display current sample information
        print(f"\nInput: ({x1}, {x2})")
        print("Actual:", y[i])
        print("Predicted:", prediction)
        print("Error:", error)

        # Update weights if prediction is wrong
        if error != 0:

            w1 = w1 + learning_rate * error * x1
            w2 = w2 + learning_rate * error * x2

            print("Weights Updated!")

        else:
            print("No Update Needed.")

        # Display updated weights
        print(f"w1 = {w1:.2f}, w2 = {w2:.2f}")

# -----------------------------
# Final Weights
# -----------------------------

print("\nFinal Weights")
print(f"w1 = {w1:.2f}")
print(f"w2 = {w2:.2f}")

# -----------------------------
# Testing and Accuracy
# -----------------------------

print("\nTesting")

# Count correct predictions
correct = 0

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

    # Display prediction and actual value
    print(f"Input ({x1:.1f}, {x2:.1f}) -> Predicted: {output}, Actual: {y[i]}")

    # Count correct predictions
    if output == y[i]:
        correct += 1

# Calculate accuracy
accuracy = (correct / len(X)) * 100

print(f"\nAccuracy: {accuracy:.2f}%")

# -----------------------------
# Visualization
# -----------------------------

# Plot data points
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="bwr",
    s=80,
    edgecolors="black"
)

# Generate x-values for decision boundary
x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Plot decision boundary
if w2 != 0:
    y_line = (threshold - w1 * x) / w2
    plt.plot(x, y_line, "k-", label="Decision Boundary")

# Labels and title
plt.xlabel("Attendance %")
plt.ylabel("Study Hours per Week")
plt.title("Perceptron Student Pass Prediction")

# Grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()