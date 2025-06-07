
# ------------------- MesoXception Training (With Validation Monitoring) -------------------

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from classifiers import *
from pipeline import *

# Initialize model
classifier = MesoXceptionNet(learning_rate=0.001)

dataset_path = "data"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20

# Data Augmentation + Validation Split
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest',
    validation_split=0.2
)

# Training Generator
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation Generator
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Learning rate scheduler (now monitoring val_loss)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

# Fit model with validation monitoring
classifier.model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler],
    verbose=1
)

# Save model weights
classifier.model.save_weights('weights/MesoXceptionNet.h5')
print("Model trained with validation monitoring and saved.")


