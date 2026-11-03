import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0


# --------------------------------------------------
# Model 1: Without Regularization
# --------------------------------------------------
model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history1 = model1.fit(x_train, y_train,
                      epochs=10,
                      validation_split=0.2)


# --------------------------------------------------
# Model 2: L2 Regularization
# --------------------------------------------------
model2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu',
          kernel_regularizer=l2(0.001)),
    Dense(10, activation='softmax')
])

model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(x_train, y_train,
                      epochs=10,
                      validation_split=0.2)


# --------------------------------------------------
# Model 3: Dropout
# --------------------------------------------------
model3 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model3.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history3 = model3.fit(x_train, y_train,
                      epochs=10,
                      validation_split=0.2)


# --------------------------------------------------
# Test Accuracy
# --------------------------------------------------
print("\nModel 1 Accuracy:")
print(model1.evaluate(x_test, y_test)[1])

print("Model 2 Accuracy:")
print(model2.evaluate(x_test, y_test)[1])

print("Model 3 Accuracy:")
print(model3.evaluate(x_test, y_test)[1])

print("\n")

# --------------------------------------------------
# Accuracy Curves
# --------------------------------------------------
plt.plot(history1.history['accuracy'], label='No Regularization')
plt.plot(history1.history['val_accuracy'], '--', label='No Reg Validation')

plt.plot(history2.history['accuracy'], label='L2')
plt.plot(history2.history['val_accuracy'], '--', label='L2 Validation')

plt.plot(history3.history['accuracy'], label='Dropout')
plt.plot(history3.history['val_accuracy'], '--', label='Dropout Validation')

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
print("\n")

# --------------------------------------------------
# Loss Curves
# --------------------------------------------------
plt.plot(history1.history['loss'], label='No Regularization')
plt.plot(history1.history['val_loss'], '--', label='No Reg Validation')

plt.plot(history2.history['loss'], label='L2')
plt.plot(history2.history['val_loss'], '--', label='L2 Validation')

plt.plot(history3.history['loss'], label='Dropout')
plt.plot(history3.history['val_loss'], '--', label='Dropout Validation')

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()