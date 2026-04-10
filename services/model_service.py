import tensorflow as tf
import numpy as np
import json
import os
import gdown

from config import AUTO_DOWNLOAD_MODEL, MODEL_PATH, CLASS_NAMES_PATH, MODEL_DOWNLOAD_URL


def _download_model_from_drive(download_url, destination):
    downloaded_path = gdown.download(
        url=download_url,
        output=destination,
        quiet=False,
        fuzzy=True,
    )
    if not downloaded_path or not os.path.exists(downloaded_path):
        raise RuntimeError("Google Drive download did not produce a local file.")


def ensure_model_available():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    if not AUTO_DOWNLOAD_MODEL:
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH} and AUTO_DOWNLOAD_MODEL is disabled."
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    temp_path = f"{MODEL_PATH}.download"

    try:
        _download_model_from_drive(MODEL_DOWNLOAD_URL, temp_path)
        os.replace(temp_path, MODEL_PATH)
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(
            "Model download failed. Set MODEL_DOWNLOAD_URL correctly or place oct_retinal_model.keras locally."
        ) from exc

    return MODEL_PATH

# Load once (important)
model = tf.keras.models.load_model(ensure_model_available())

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    
    return class_names[predicted_idx], predictions, img_array
