import tensorflow as tf
import cv2
import numpy as np

# Define class labels
class_labels = ['calico', 'generic_cat', 'persian']

def preprocess_image(image_path):
    """Preprocess the input image."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0  # Normalize to [0, 1]
    return image

def predict_image(model, image_path):
    """Predict the class of the input image using the model."""
    # Preprocess the image
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, predictions

# Load the model
model = tf.keras.models.load_model('/Users/moka/Documents/cats/saved_models/cat_classifier_model.h5')

# Path to a new image
new_image_path = '/Users/moka/Documents/cats/predict_cat.jpg'

# Make prediction
predicted_class_label, predictions = predict_image(model, new_image_path)

# Print the result
print(f'Predicted class: {predicted_class_label}')
print(f'Predictions: {predictions}')
