from flask import Blueprint, request, jsonify
import os
import numpy as np

from config import UPLOAD_FOLDER, OUTPUT_FOLDER
from services.model_service import predict_image, class_names, model
from services.gradcam_service import get_gradcam_heatmap
from services.fused_gradcam import run_full_pipeline
from services.plot_service import save_probability_plot
from services.scan_chat_service import ask_scan_chat
from services.visualize_results import save_visualization_results
from utils.file_utils import save_uploaded_file

import tensorflow as tf

predict_bp = Blueprint("predict", __name__)


def _absolute_url(path):
    return request.host_url.rstrip("/") + path

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
        "image_url": _absolute_url("/static/uploads/" + filename),
        "plot_url": _absolute_url("/static/uploads/" + "plot_" + filename),
        "cam_url": _absolute_url("/static/uploads/" + "cam_" + filename) if heatmap is not None else None
    })


@predict_bp.route("/analyze_fused", methods=["POST"])
def analyze_fused():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename, filepath = save_uploaded_file(file)

    result = run_full_pipeline(
        model=model,
        img_path=filepath,
        class_labels=class_names,
        preprocess_fn=tf.keras.applications.resnet50.preprocess_input,
    )
    saved_paths = save_visualization_results(result, output_dir=OUTPUT_FOLDER)

    return jsonify({
        "predicted_class": result["predicted_class"],
        "original_confidence": result["original_confidence"],
        "adaptive_weights": result["adaptive_weights"],
        "adaptive_strategy": result["adaptive_strategy"],
        "validation": {
            "layer2": result["validation"]["layer2_mask_confidence"],
            "layer3": result["validation"]["layer3_mask_confidence"],
            "layer4": result["validation"]["layer4_mask_confidence"],
            "fused": result["validation"]["fused_mask_confidence"],
        },
        "result_image_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_result_path"])),
        "standard_gradcam_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_standard_overlay_path"])),
        "fused_cam_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_fused_overlay_path"])),
        "layer2_heatmap_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_layer2_heatmap_path"])),
        "layer3_heatmap_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_layer3_heatmap_path"])),
        "layer4_heatmap_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_layer4_heatmap_path"])),
        "fused_heatmap_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_fused_heatmap_path"])),
        "binary_mask_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_binary_mask_path"])),
        "masked_oct_url": _absolute_url("/output/" + os.path.basename(saved_paths["latest_masked_oct_path"])),
        "uploaded_image": _absolute_url("/static/uploads/" + filename),
    })


@predict_bp.route("/chat_scan", methods=["POST"])
def chat_scan():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    analysis = payload.get("analysis") or {}

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        answer = ask_scan_chat(question, analysis)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Scan chatbot failed to generate a response."}), 500

    return jsonify({"answer": answer})
