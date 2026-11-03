# =========================================================
# VISUALIZE AND COMPARE ACTIVATION FUNCTIONS
# Sigmoid vs Tanh vs ReLU
# =========================================================


# ---------------------------------------------------------
# 1. Import libraries
# ---------------------------------------------------------

import tensorflow as tf
import matplotlib.pyplot as plt
import time

print("\n")


# ---------------------------------------------------------
# 2. Load MNIST dataset
# ---------------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

print("MNIST dataset loaded successfully.")
print("\n")


# ---------------------------------------------------------
# 3. Create and train Sigmoid model
# ---------------------------------------------------------

sigmoid_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

sigmoid_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start_time = time.time()

sigmoid_history = sigmoid_model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

sigmoid_time = time.time() - start_time

# Test the model
sigmoid_test_loss, sigmoid_test_accuracy = sigmoid_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n")
print("========== SIGMOID RESULTS ==========")

print("Training time:",
      round(sigmoid_time, 2), "seconds")

print("Training accuracy:",
      round(sigmoid_history.history['accuracy'][-1] * 100, 2), "%")

print("Validation accuracy:",
      round(sigmoid_history.history['val_accuracy'][-1] * 100, 2), "%")

print("Test accuracy:",
      round(sigmoid_test_accuracy * 100, 2), "%")

print("Final training loss:",
      round(sigmoid_history.history['loss'][-1], 4))

print("Final validation loss:",
      round(sigmoid_history.history['val_loss'][-1], 4))

print("\n")


# ---------------------------------------------------------
# 4. Create and train Tanh model
# ---------------------------------------------------------

tanh_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax')
])

tanh_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start_time = time.time()

tanh_history = tanh_model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

tanh_time = time.time() - start_time

# Test the model
tanh_test_loss, tanh_test_accuracy = tanh_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n")
print("========== TANH RESULTS ==========")

print("Training time:",
      round(tanh_time, 2), "seconds")

print("Training accuracy:",
      round(tanh_history.history['accuracy'][-1] * 100, 2), "%")

print("Validation accuracy:",
      round(tanh_history.history['val_accuracy'][-1] * 100, 2), "%")

print("Test accuracy:",
      round(tanh_test_accuracy * 100, 2), "%")

print("Final training loss:",
      round(tanh_history.history['loss'][-1], 4))

print("Final validation loss:",
      round(tanh_history.history['val_loss'][-1], 4))

print("\n")


# ---------------------------------------------------------
# 5. Create and train ReLU model
# ---------------------------------------------------------

relu_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

relu_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start_time = time.time()

relu_history = relu_model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

relu_time = time.time() - start_time

# Test the model
relu_test_loss, relu_test_accuracy = relu_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n")
print("========== RELU RESULTS ==========")

print("Training time:",
      round(relu_time, 2), "seconds")

print("Training accuracy:",
      round(relu_history.history['accuracy'][-1] * 100, 2), "%")

print("Validation accuracy:",
      round(relu_history.history['val_accuracy'][-1] * 100, 2), "%")

print("Test accuracy:",
      round(relu_test_accuracy * 100, 2), "%")

print("Final training loss:",
      round(relu_history.history['loss'][-1], 4))

print("Final validation loss:",
      round(relu_history.history['val_loss'][-1], 4))

print("\n")


# ---------------------------------------------------------
# 6. Calculate gradients
# ---------------------------------------------------------

def get_gradients(model):

    # Take first 100 training images
    x_sample = x_train[:100]
    y_sample = y_train[:100]

    # Calculate gradients
    with tf.GradientTape() as tape:

        predictions = model(x_sample, training=True)

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_sample,
            predictions
        )

        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    # Gradient of first hidden layer
    early_gradient = tf.reduce_mean(
        tf.abs(gradients[0])
    ).numpy()

    # Gradient of second hidden layer
    later_gradient = tf.reduce_mean(
        tf.abs(gradients[2])
    ).numpy()

    return early_gradient, later_gradient


# Calculate gradients for all three models

sigmoid_early, sigmoid_later = get_gradients(
    sigmoid_model
)

tanh_early, tanh_later = get_gradients(
    tanh_model
)

relu_early, relu_later = get_gradients(
    relu_model
)


# Print gradient results

print("========== GRADIENT RESULTS ==========")

print("Sigmoid early layer gradient:",
      format(sigmoid_early, ".6f"))

print("Sigmoid later layer gradient:",
      format(sigmoid_later, ".6f"))

print("\n")

print("Tanh early layer gradient:",
      format(tanh_early, ".6f"))

print("Tanh later layer gradient:",
      format(tanh_later, ".6f"))

print("\n")

print("ReLU early layer gradient:",
      format(relu_early, ".6f"))

print("ReLU later layer gradient:",
      format(relu_later, ".6f"))

print("\n")


# ---------------------------------------------------------
# 7. Check inactive ReLU neurons
# ---------------------------------------------------------

# This check is only needed for ReLU because
# ReLU can have inactive/dead neurons.

# Create a model that gives the output of the
# first ReLU hidden layer

relu_hidden_model = tf.keras.Model(
    inputs=relu_model.inputs[0],
    outputs=relu_model.layers[2].output
)

# Get hidden-layer outputs for 1000 training images

hidden_output = relu_hidden_model.predict(
    x_train[:1000],
    verbose=0
)

# Count how many times each neuron gives a positive output

positive_count = (hidden_output > 0).sum(axis=0)

# A neuron is inactive if it never becomes positive

inactive_neurons = (positive_count == 0).sum()

print("========== INACTIVE RELU NEURON CHECK ==========")

print("Number of inactive ReLU neurons:",
      inactive_neurons)

print("\n")


# ---------------------------------------------------------
# 8. Plot Validation Accuracy
# ---------------------------------------------------------

plt.figure()

plt.plot(
    sigmoid_history.history['val_accuracy'],
    label='Sigmoid'
)

plt.plot(
    tanh_history.history['val_accuracy'],
    label='Tanh'
)

plt.plot(
    relu_history.history['val_accuracy'],
    label='ReLU'
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")

plt.title("Validation Accuracy Comparison")

plt.legend()

plt.show()

print("\n")


# ---------------------------------------------------------
# 9. Plot Validation Loss
# ---------------------------------------------------------

plt.figure()

plt.plot(
    sigmoid_history.history['val_loss'],
    label='Sigmoid'
)

plt.plot(
    tanh_history.history['val_loss'],
    label='Tanh'
)

plt.plot(
    relu_history.history['val_loss'],
    label='ReLU'
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

plt.title("Validation Loss Comparison")

plt.legend()

plt.show()

print("\n")


# ---------------------------------------------------------
# 10. Plot Training Time
# ---------------------------------------------------------

activation_names = [
    'Sigmoid',
    'Tanh',
    'ReLU'
]

training_times = [
    sigmoid_time,
    tanh_time,
    relu_time
]

plt.figure()

plt.bar(
    activation_names,
    training_times
)

plt.xlabel("Activation Function")
plt.ylabel("Training Time (seconds)")

plt.title("Training Time Comparison")

plt.show()

print("\n")


# ---------------------------------------------------------
# 11. Plot Gradient Flow
# ---------------------------------------------------------

early_gradients = [
    sigmoid_early,
    tanh_early,
    relu_early
]

later_gradients = [
    sigmoid_later,
    tanh_later,
    relu_later
]

plt.figure()

plt.plot(
    activation_names,
    early_gradients,
    marker='o',
    label='Early Layer'
)

plt.plot(
    activation_names,
    later_gradients,
    marker='o',
    label='Later Layer'
)

# Log scale makes small gradients easier to see

plt.yscale('log')

plt.xlabel("Activation Function")
plt.ylabel("Gradient Magnitude")

plt.title("Gradient Flow Comparison")

plt.legend()

plt.show()

print("\n")


# ---------------------------------------------------------
# 12. Final comparison output
# ---------------------------------------------------------

print("========== FINAL COMPARISON ==========")

print("\n")

print("Training Time:")
print("Sigmoid:", round(sigmoid_time, 2), "seconds")
print("Tanh:", round(tanh_time, 2), "seconds")
print("ReLU:", round(relu_time, 2), "seconds")

print("\n")

print("Training Accuracy:")
print(
    "Sigmoid:",
    round(sigmoid_history.history['accuracy'][-1] * 100, 2),
    "%"
)
print(
    "Tanh:",
    round(tanh_history.history['accuracy'][-1] * 100, 2),
    "%"
)
print(
    "ReLU:",
    round(relu_history.history['accuracy'][-1] * 100, 2),
    "%"
)

print("\n")

print("Validation Accuracy:")
print(
    "Sigmoid:",
    round(sigmoid_history.history['val_accuracy'][-1] * 100, 2),
    "%"
)
print(
    "Tanh:",
    round(tanh_history.history['val_accuracy'][-1] * 100, 2),
    "%"
)
print(
    "ReLU:",
    round(relu_history.history['val_accuracy'][-1] * 100, 2),
    "%"
)

print("\n")

print("Test Accuracy:")
print(
    "Sigmoid:",
    round(sigmoid_test_accuracy * 100, 2),
    "%"
)
print(
    "Tanh:",
    round(tanh_test_accuracy * 100, 2),
    "%"
)
print(
    "ReLU:",
    round(relu_test_accuracy * 100, 2),
    "%"
)

print("\n")

print("Final Training Loss:")
print(
    "Sigmoid:",
    round(sigmoid_history.history['loss'][-1], 4)
)
print(
    "Tanh:",
    round(tanh_history.history['loss'][-1], 4)
)
print(
    "ReLU:",
    round(relu_history.history['loss'][-1], 4)
)

print("\n")

print("Final Validation Loss:")
print(
    "Sigmoid:",
    round(sigmoid_history.history['val_loss'][-1], 4)
)
print(
    "Tanh:",
    round(tanh_history.history['val_loss'][-1], 4)
)
print(
    "ReLU:",
    round(relu_history.history['val_loss'][-1], 4)
)

print("\n")

print("Gradient Magnitude:")
print("Sigmoid early:", format(sigmoid_early, ".6f"))
print("Sigmoid later:", format(sigmoid_later, ".6f"))

print("\n")

print("Tanh early:", format(tanh_early, ".6f"))
print("Tanh later:", format(tanh_later, ".6f"))

print("\n")

print("ReLU early:", format(relu_early, ".6f"))
print("ReLU later:", format(relu_later, ".6f"))

print("\n")

print("Inactive ReLU neurons:",
      inactive_neurons)

print("\n")

print("========== PROGRAM COMPLETED ==========")

print("\n")