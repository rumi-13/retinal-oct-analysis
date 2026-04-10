from flask import Blueprint, request, jsonify
import os
import numpy as np

from config import UPLOAD_FOLDER, BASE_URL
from services.model_service import predict_image, class_names
from services.gradcam_service import get_gradcam_heatmap
from services.plot_service import save_probability_plot
from utils.file_utils import save_uploaded_file

import tensorflow as tf

predict_bp = Blueprint("predict", __name__)

def save_gradcam_image(img_path, heatmap, output_path, alpha=0.4):
    import numpy as np
    import matplotlib

    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(output_path)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    filename, filepath = save_uploaded_file(file)

    pred_class, probabilities, img_array = predict_image(filepath)
    confidence = float(probabilities[np.argmax(probabilities)])

    plot_path = os.path.join(UPLOAD_FOLDER, "plot_" + filename)
    save_probability_plot(class_names, probabilities, plot_path)

    heatmap = get_gradcam_heatmap(img_array)
    cam_path = os.path.join(UPLOAD_FOLDER, "cam_" + filename)

    if heatmap is not None:
        save_gradcam_image(filepath, heatmap, cam_path)

    return jsonify({
        "prediction": pred_class,
        "confidence_score": confidence,
        "confidence": f"{confidence:.2%}",
        "image_url": BASE_URL + "static/uploads/" + filename,
        "plot_url": BASE_URL + "static/uploads/" + "plot_" + filename,
        "cam_url": BASE_URL + "static/uploads/" + "cam_" + filename if heatmap is not None else None
    })
