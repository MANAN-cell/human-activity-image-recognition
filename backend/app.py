
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "human_activity_image_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "BENDING",
    "JUMPING",
    "LYING",
    "RUNNING",
    "SITTING",
    "SQUATTING",
    "STANDING",
    "STRETCHING",
    "WALKING"
]

@app.route("/")
def home():
    return "Human Activity Recognition API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        img = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

        img = img.resize((160, 160))

        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr, verbose=0)[0]

        idx = int(np.argmax(predictions))

        return jsonify({
            "activity": class_names[idx],
            "confidence": round(
                float(predictions[idx]) * 100, 2
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
