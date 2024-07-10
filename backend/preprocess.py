import cv2
import numpy as np
from albumentations import Compose, RandomRotate90, Flip, Transpose

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize to [0, 1]
    return image

augmentation = Compose([
    RandomRotate90(),
    Flip(),
    Transpose()
])

def augment_image(image):
    augmented = augmentation(image=image)
    return augmented['image']
