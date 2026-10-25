"""
Purpose:  Load the image datasets, build MobileNetV2, and train the model.
This script saves the trained model and training history in Outputs/.
Required packages: TensorFlow
"""

from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR = PROJECT_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_DIR = DATA_DIR / "Split_Datasets" / "train"
TEST_DIR = DATA_DIR / "Split_Datasets" / "test"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Coccinella_novemnotata", "Harmonia_axyridis"]
BATCH_SIZE = 16

# Use the same seed and class order when loading images.
tf.random.set_seed(SEED)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

,
)

# Prefetch batches to reduce waiting while the model is training.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Class mapping:", dict(enumerate(CLASS_NAMES)))

# Load MobileNetV2 without its original ImageNet classification head.
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # Keep pretrained feature weights frozen.

# Build a binary classifier on top of the pretrained feature extractor.
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    # MobileNetV2 expects pixel values scaled from 0–255 to -1–1.
    layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# Stop early if validation loss does not improve for three epochs.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping],
)

# Save model and epoch-by-epoch history for documentation/reuse.
model.save(OUTPUT_DIR / "spotcheck_mobilenetv2.keras")
with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as file:
    json.dump(history.history, file, indent=2)

print("Model and training history saved to:", OUTPUT_DIR)
