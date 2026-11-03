# Dataset: https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time

# 1. Load and split dataset
train = tf.keras.utils.image_dataset_from_directory(
    "data", image_size=(224, 224), batch_size=16,
    validation_split=0.2, subset="training", seed=1
)

val = tf.keras.utils.image_dataset_from_directory(
    "data", image_size=(224, 224), batch_size=16,
    validation_split=0.2, subset="validation", seed=1
)

# 2. Data augmentation
augment = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

# 3. Load pretrained VGG16 base
base = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

# 4. Create the new classifier
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    augment,
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# 5. Experiment A: Feature extraction
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

start = time.time()
h1 = model.fit(train, validation_data=val, epochs=3)
print("Feature extraction time:", time.time() - start)

# 6. Experiment B: Fine-tuning
base.trainable = True

for layer in base.layers[:15]:
    layer.trainable = False

for layer in base.layers[15:]:
    layer.trainable = False

# Unfreeze only the final convolutional block
for layer in base.layers:
    if "block5" in layer.name:
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

start = time.time()
h2 = model.fit(train, validation_data=val, epochs=3)
print("Fine-tuning time:", time.time() - start)

# 7. Compare accuracy
plt.plot(range(1, 4), h1.history["accuracy"], marker="o",
         label="Feature Extraction")
plt.plot(range(1, 4), h2.history["accuracy"], marker="o",
         label="Fine-Tuning")
plt.plot(range(1, 4), h1.history["val_accuracy"], marker="o",
         label="Validation A")
plt.plot(range(1, 4), h2.history["val_accuracy"], marker="o",
         label="Validation B")

plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks([1, 2, 3])
plt.legend()
plt.grid(True)
plt.show()

# 8. Compare loss
plt.plot(range(1, 4), h1.history["loss"], marker="o",
         label="Feature Extraction")
plt.plot(range(1, 4), h2.history["loss"], marker="o",
         label="Fine-Tuning")
plt.plot(range(1, 4), h1.history["val_loss"], marker="o",
         label="Validation A")
plt.plot(range(1, 4), h2.history["val_loss"], marker="o",
         label="Validation B")

plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks([1, 2, 3])
plt.legend()
plt.grid(True)
plt.show()

# 9. Save model
model.save("cat_dog_model.keras")
print("Model saved!")