"""
Purpose: Evaluate the trained model and create model-performance plots.

Outputs:
- Outputs/confusion_matrix.png
- Outputs/training_validation_accuracy.png
- Outputs/training_validation_loss.png
- Outputs/classification_report.txt
- Outputs/test_predictions.csv

Required packages: TensorFlow, matplotlib, scikit-learn, pandas, numpy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR = PROJECT_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["Coccinella_novemnotata", "Harmonia_axyridis"]
TEST_DIR = DATA_DIR / "Split_Datasets" / "test"

# Standalone evaluation: load the saved model and test data.
model = tf.keras.models.load_model(OUTPUT_DIR / "spotcheck_mobilenetv2.keras")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Recover the saved training history for accuracy/loss plots.
import json
with open(OUTPUT_DIR / "training_history.json", "r", encoding="utf-8") as file:
    history = json.load(file)

# Get the true class for each test image and the model's probability estimates.
y_true = np.concatenate([labels.numpy() for images, labels in test_ds])
y_prob = model.predict(test_ds, verbose=0)
y_pred = (y_prob.ravel() >= 0.5).astype(int)

test_accuracy = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0,
)
print(f"Test accuracy: {test_accuracy:.2%}")
print("\nClassification report:\n", report)

with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
    file.write(f"Test accuracy: {test_accuracy:.2%}\n\n")
    file.write(report)

# Save predictions with their true and predicted class labels.
predictions = pd.DataFrame({
    "true_label": [CLASS_NAMES[int(label)] for label in y_true],
    "predicted_label": [CLASS_NAMES[int(label)] for label in y_pred],
    "probability_harmonia_axyridis": y_prob.ravel(),
})
predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# Plot the confusion matrix. Rows are true labels; columns are predictions.
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title("MobileNetV2: Test Set Confusion Matrix")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot training and validation accuracy across epochs.
epochs = range(1, len(history["accuracy"]) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("MobileNetV2: Training vs. Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_validation_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot training and validation loss across epochs.
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["loss"], marker="o", label="Training Loss")
plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
plt.title("MobileNetV2: Training vs. Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_validation_loss.png", dpi=300, bbox_inches="tight")
plt.show()

print("Evaluation outputs saved to:", OUTPUT_DIR)
