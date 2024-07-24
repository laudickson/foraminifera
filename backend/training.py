import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
from albumentations import Compose, RandomRotate90, Flip, Transpose

# Number of classes in your dataset
number_of_classes = 3  # Update this with the actual number of classes

# Paths to your dataset
train_dataset_path = '/Users/moka/Documents/cats/train'
validation_dataset_path = '/Users/moka/Documents/cats/validation'

# Check if the dataset paths exist
if not os.path.exists(train_dataset_path):
    raise ValueError(f"Train dataset path {train_dataset_path} does not exist")
if not os.path.exists(validation_dataset_path):
    raise ValueError(f"Validation dataset path {validation_dataset_path} does not exist")

# Augmentation function
augmentation = Compose([
    RandomRotate90(),
    Flip(),
    Transpose()
])

def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image / 255.0  # Normalize to [0, 1]
    augmented = augmentation(image=image)
    return augmented['image']

def simple_preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Data preparation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=simple_preprocess_image  # Just preprocess, no augmentation
)

# Train generator
train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Check if both generators found images
if train_generator.samples == 0:
    raise ValueError("No training images found. Check your dataset directory structure.")
if validation_generator.samples == 0:
    raise ValueError("No validation images found. Check your dataset directory structure.")

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Ensure the directory exists
os.makedirs('/Users/moka/Documents/cats/saved_models', exist_ok=True)

# Save the model
model.save('/Users/moka/Documents/cats/saved_models/cat_classifier_model.h5')
