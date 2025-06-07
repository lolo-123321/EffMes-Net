#------------put in 'axample.py' file----------------------------
#----------------new training code--------------------------

#--------------------training code-----------------------------
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classifiers import *
from pipeline import *

# Initialize and train the model
classifier = MesoEfficientNet(learning_rate=0.001)

dataset_path = "data"  # Update with actual dataset path
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# Load training images using ImageDataGenerator
dataGenerator = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
train_generator = dataGenerator.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)
val_generator = dataGenerator.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Train the model using dataset
epochs = 20
classifier.model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save the trained model weights
classifier.model.save_weights('weights/MesoEfficientNet.h5')
print("New weights saved successfully!")
