import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# ---------------- MLP ----------------

mlp = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

mlp.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

start = time.time()

history_mlp = mlp.fit(x_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.1)

mlp_time = time.time() - start


# ---------------- CNN ----------------

cnn = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

start = time.time()

history_cnn = cnn.fit(x_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.1)

cnn_time = time.time() - start


# ---------------- Test Results ----------------

mlp_loss, mlp_acc = mlp.evaluate(x_test, y_test, verbose=0)
cnn_loss, cnn_acc = cnn.evaluate(x_test, y_test, verbose=0)

print("\n========== RESULTS ==========")

print("\nMLP")
print("Test Accuracy:", mlp_acc)
print("Test Loss:", mlp_loss)
print("Training Time:", mlp_time)

print("\nCNN")
print("Test Accuracy:", cnn_acc)
print("Test Loss:", cnn_loss)
print("Training Time:", cnn_time)

print("\n")


# ---------------- Accuracy Graph ----------------

plt.plot(history_mlp.history['val_accuracy'], label='MLP')
plt.plot(history_cnn.history['val_accuracy'], label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('MLP vs CNN Accuracy')
plt.legend()
plt.show()
print("\n")


# ---------------- Loss Graph ----------------

plt.plot(history_mlp.history['val_loss'], label='MLP')
plt.plot(history_cnn.history['val_loss'], label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('MLP vs CNN Loss')
plt.legend()
plt.show()
print("\n")


# ---------------- MLP Predictions ----------------

predictions_mlp = mlp.predict(x_test[:10])

plt.figure(figsize=(15, 8))

for i in range(10):
    predicted_class = class_names[tf.argmax(predictions_mlp[i])]
    actual_class = class_names[y_test[i][0]]

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title("Actual: " + actual_class +
              "\nMLP: " + predicted_class)
    plt.axis('off')

plt.suptitle("MLP Predictions")
plt.tight_layout()
plt.show()
print("\n")


# ---------------- CNN Predictions ----------------

predictions_cnn = cnn.predict(x_test[:10])

plt.figure(figsize=(15, 8))

for i in range(10):
    predicted_class = class_names[tf.argmax(predictions_cnn[i])]
    actual_class = class_names[y_test[i][0]]

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title("Actual: " + actual_class +
              "\nCNN: " + predicted_class)
    plt.axis('off')

plt.suptitle("CNN Predictions")
plt.tight_layout()
plt.show()