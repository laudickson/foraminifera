from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model("foraminifera_model.h5")

IMAGE_DIR = "path_to_save_image/"
os.makedirs(IMAGE_DIR, exist_ok=True)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_path = os.path.join(IMAGE_DIR, file.filename)
    file.save(img_path)
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    return jsonify({"predicted_class": str(predicted_class[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
