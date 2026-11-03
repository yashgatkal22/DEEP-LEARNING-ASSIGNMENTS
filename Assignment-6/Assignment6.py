# Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/IMDB.csv")
X = data["review"].astype(str)
y = data["sentiment"].map({"positive": 1, "negative": 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save original test reviews
X_test_original = X_test.copy()

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile and train
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=3, batch_size=128, validation_split=0.2)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc * 100, "%")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("LSTM Model Accuracy")
plt.legend()
plt.show()

# Predict 5 example reviews
predictions = model.predict(X_test[:5])

for i in range(5):
    print("\nReview:", X_test_original.iloc[i][:200])
    print("Predicted:", "Positive" if predictions[i][0] >= 0.5 else "Negative")
    print("Actual:", "Positive" if y_test.iloc[i] == 1 else "Negative")