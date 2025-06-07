#------------ Predict on Test Set --------------------

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classifiers import *
from pipeline import *

# Load trained model
classifier = MesoXceptionNet(learning_rate=0.001)
classifier.load('weights/MesoXceptionNet.h5')
print("Loaded pre-trained weights!")

# Paths and constants
dataset_path = "data"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# --- ImageDataGenerator for both train and test (same 80/20 split) ---
data_gen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

# TRAIN set (used during training)
train_gen = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=1,  # Needed to access individual filenames
    class_mode='binary',
    shuffle=False,
    subset='training'
)

# TEST set (used only now)
test_gen = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

# Overlap check
train_files = set(train_gen.filenames)
test_files = set(test_gen.filenames)
overlap = train_files & test_files

if len(overlap) == 0:
    print("No overlap between training and test sets — validation is clean.")
else:
    print(f"WARNING: Overlap detected! {len(overlap)} shared files.")
    print(list(overlap)[:5])  # Print a few overlaps for review

# Reload TEST generator with appropriate batch size
test_gen = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

# --- Predict ---
print("\n🔍 Predicting on the test set...\n")
probs = classifier.model.predict(test_gen, verbose=1).ravel()
true_labels = test_gen.classes
filenames = test_gen.filenames
predictions = (probs > 0.5).astype(int)

# Print predictions
for i in range(len(filenames)):
    confidence = probs[i] if predictions[i] == 1 else 1 - probs[i]
    print(f"Image: {filenames[i]} | Predicted: {'Real' if predictions[i] == 1 else 'Fake'} "
          f"| Confidence: {confidence:.4f} | Actual: {'Real' if true_labels[i] == 1 else 'Fake'}")

# --- Evaluation ---
acc = accuracy_score(true_labels, predictions)
print(f"\n Test Set Accuracy: {acc:.4f}")
print("\n Classification Report:")
print(classification_report(true_labels, predictions, target_names=["Fake", "Real"]))

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=300)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='orange', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("test_roc_curve.png", dpi=300)
plt.show()
