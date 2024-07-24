import tensorflow as tf
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('cat_classifier_model.h5')

def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['file']
    img_path = f"./temp/{img_file.filename}"
    img_file.save(img_path)
    img_array = prepare_image(img_path)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
