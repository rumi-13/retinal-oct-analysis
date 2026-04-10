import tensorflow as tf
import numpy as np
import json
import os
import re
from urllib.request import HTTPCookieProcessor, build_opener

from config import AUTO_DOWNLOAD_MODEL, MODEL_PATH, CLASS_NAMES_PATH, MODEL_DOWNLOAD_URL


def _save_response_content(response, destination):
    with open(destination, "wb") as file_obj:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)


def _extract_confirm_token(opener, html):
    for cookie in opener.handlers[0].cookiejar:
        if cookie.name.startswith("download_warning"):
            return cookie.value

    patterns = [
        r'confirm=([0-9A-Za-z_]+)',
        r'name="confirm"\s+value="([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None


def _download_model_from_drive(download_url, destination):
    opener = build_opener(HTTPCookieProcessor())

    with opener.open(download_url) as response:
        content_type = response.headers.get("Content-Type", "")
        content_disposition = response.headers.get("Content-Disposition", "")

        if "application/octet-stream" in content_type or "attachment" in content_disposition.lower():
            _save_response_content(response, destination)
            return

        html = response.read().decode("utf-8", errors="ignore")

    confirm_token = _extract_confirm_token(opener, html)
    if not confirm_token:
        raise RuntimeError("Could not resolve Google Drive download confirmation token.")

    separator = "&" if "?" in download_url else "?"
    confirmed_url = f"{download_url}{separator}confirm={confirm_token}"

    with opener.open(confirmed_url) as response:
        content_disposition = response.headers.get("Content-Disposition", "")
        if "attachment" not in content_disposition.lower():
            final_url = response.geturl()
            if final_url != confirmed_url:
                with opener.open(final_url) as redirected_response:
                    _save_response_content(redirected_response, destination)
                    return
        _save_response_content(response, destination)


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
