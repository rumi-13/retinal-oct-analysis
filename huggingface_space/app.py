import json
import os
import tempfile

import gdown
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

matplotlib.use("Agg")


MODEL_FILE_ID = "1jXrsLBiQHOnukSr1dg3CIeUXFshr05cF"
MODEL_DOWNLOAD_URL = os.environ.get(
    "MODEL_DOWNLOAD_URL",
    f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}",
)
MODEL_PATH = "oct_retinal_model.keras"
CLASS_NAMES_PATH = "class_names.json"
DEFAULT_CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

DISEASE_DETAILS = {
    "CNV": {
        "name": "Choroidal Neovascularization",
        "description": "Abnormal blood vessel growth beneath the retina that can leak fluid or blood and threaten central vision.",
    },
    "DME": {
        "name": "Diabetic Macular Edema",
        "description": "Swelling in the macula caused by leaking retinal blood vessels, commonly associated with diabetic retinopathy.",
    },
    "DRUSEN": {
        "name": "Drusen Deposits",
        "description": "Yellow extracellular deposits beneath the retina that are commonly discussed in age-related macular degeneration assessment.",
    },
    "NORMAL": {
        "name": "No major abnormality detected",
        "description": "The scan most closely matches the non-pathological class learned during training.",
    },
}


def ensure_model_available():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    temp_path = f"{MODEL_PATH}.download"
    downloaded_path = gdown.download(
        url=MODEL_DOWNLOAD_URL,
        output=temp_path,
        quiet=False,
        fuzzy=True,
    )

    if not downloaded_path or not os.path.exists(downloaded_path):
        raise RuntimeError("Model download failed. Please check MODEL_DOWNLOAD_URL.")

    os.replace(temp_path, MODEL_PATH)
    return MODEL_PATH


def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    return DEFAULT_CLASS_NAMES


MODEL = tf.keras.models.load_model(ensure_model_available())
CLASS_NAMES = load_class_names()


def predict_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)

    predictions = MODEL.predict(image_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    return CLASS_NAMES[predicted_idx], predictions, image_array


def get_gradcam_heatmap(img_array, last_conv_layer_name="conv5_block3_out"):
    base_model = next((layer for layer in MODEL.layers if isinstance(layer, tf.keras.Model)), None)
    if not base_model:
        return None

    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    base_model_idx = MODEL.layers.index(base_model)
    for layer in MODEL.layers[base_model_idx + 1 :]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if float(max_val) <= 0:
        return None
    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()


def create_heatmap_overlay(original_path, heatmap):
    image = Image.open(original_path).convert("RGB")
    image_array = np.array(image)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_image = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(image.size)
    jet_array = np.array(jet_image)

    overlay = np.uint8(np.clip(0.4 * jet_array + 0.6 * image_array, 0, 255))
    return Image.fromarray(overlay)


def create_probability_plot(probabilities):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, probabilities, color=["#34d399", "#22d3ee", "#fbbf24", "#a78bfa"])
    ax.set_title("Class Probabilities")
    ax.set_ylabel("Confidence")
    ax.set_ylim([0, 1])
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.0%}",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontsize=9,
        )
    plt.tight_layout()
    return fig


def build_confidence_card(prediction, confidence, probabilities):
    detail = DISEASE_DETAILS.get(prediction, {"name": prediction, "description": "Model class detail unavailable."})
    bars_html = "".join(
        f"""
        <div class="prob-row">
          <span>{label}</span>
          <div class="prob-track"><div class="prob-fill" style="width:{score * 100:.1f}%"></div></div>
          <strong>{score:.1%}</strong>
        </div>
        """
        for label, score in zip(CLASS_NAMES, probabilities)
    )
    return f"""
    <div class="result-card">
      <div class="eyebrow">Disease Prediction</div>
      <h2>{prediction}</h2>
      <p class="detail-title">{detail['name']}</p>
      <p class="body-copy">{detail['description']}</p>
      <div class="confidence-head">
        <span>Confidence Level</span>
        <strong>{confidence:.2%}</strong>
      </div>
      <div class="confidence-track"><div class="confidence-fill" style="width:{confidence * 100:.1f}%"></div></div>
      <div class="prob-grid">{bars_html}</div>
    </div>
    """


def analyze_image(upload):
    if upload is None:
        raise gr.Error("Please upload a retinal OCT image first.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image = Image.fromarray(upload.astype(np.uint8)).convert("RGB")
        image.save(temp_file.name)
        temp_path = temp_file.name

    try:
        prediction, probabilities, img_array = predict_image(temp_path)
        confidence = float(probabilities[np.argmax(probabilities)])
        heatmap = get_gradcam_heatmap(img_array)
        overlay = create_heatmap_overlay(temp_path, heatmap) if heatmap is not None else None
        plot = create_probability_plot(probabilities)
        result_html = build_confidence_card(prediction, confidence, probabilities)
        return result_html, image, overlay, plot
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


CUSTOM_CSS = """
body, .gradio-container {
  background: radial-gradient(circle at top, #123047 0%, #08111e 48%, #050913 100%) !important;
  color: #e2e8f0 !important;
  font-family: "Segoe UI", sans-serif;
}
.gradio-container {
  max-width: 1200px !important;
}
.top-shell, .content-card, .result-card, .upload-card, .info-card, .workflow-card {
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 28px;
  box-shadow: 0 28px 80px rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(18px);
}
.top-shell {
  padding: 28px;
  margin-bottom: 18px;
}
.brand-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  border-radius: 999px;
  background: rgba(34, 211, 238, 0.12);
  color: #d1fae5;
  border: 1px solid rgba(34, 211, 238, 0.2);
  font-size: 14px;
  margin-bottom: 18px;
}
.hero-title {
  font-size: 44px;
  line-height: 1.02;
  color: #f8fafc;
  margin: 0 0 14px 0;
}
.hero-copy, .body-copy {
  color: #cbd5e1;
  line-height: 1.8;
}
.section-title {
  font-size: 34px;
  color: #f8fafc;
  margin: 0 0 10px 0;
}
.eyebrow {
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #6ee7b7;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 12px;
}
.content-card, .upload-card {
  padding: 24px;
}
.info-grid, .feature-grid, .workflow-grid {
  display: grid;
  gap: 16px;
}
.feature-grid {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}
.workflow-grid {
  grid-template-columns: 1fr;
}
.info-card, .workflow-card {
  padding: 18px;
}
.class-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}
.class-badge {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.05);
  color: #f8fafc;
}
.result-card {
  padding: 24px;
}
.result-card h2 {
  font-size: 34px;
  color: #f8fafc;
  margin: 6px 0 10px 0;
}
.detail-title {
  color: #67e8f9;
  font-weight: 600;
  margin-bottom: 12px;
}
.confidence-head, .prob-row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 12px;
  align-items: center;
}
.confidence-head {
  margin-top: 18px;
  margin-bottom: 10px;
}
.confidence-track, .prob-track {
  width: 100%;
  background: rgba(255,255,255,0.1);
  border-radius: 999px;
  overflow: hidden;
}
.confidence-track {
  height: 12px;
}
.prob-track {
  height: 10px;
}
.confidence-fill, .prob-fill {
  height: 100%;
  background: linear-gradient(90deg, #22d3ee, #34d399, #a3e635);
}
.prob-grid {
  margin-top: 18px;
  display: grid;
  gap: 10px;
}
.warning-box {
  padding: 18px 20px;
  border-radius: 22px;
  border: 1px solid rgba(251, 191, 36, 0.25);
  background: rgba(251, 191, 36, 0.12);
  color: #fef3c7;
  margin-bottom: 18px;
}
"""


HOME_HTML = """
<div class="top-shell">
  <div class="brand-pill">Ophthalmic AI Screening Interface</div>
  <h1 class="hero-title">Turn retinal OCT scans into explainable clinical insights.</h1>
  <p class="hero-copy">
    This project combines deep learning, retinal OCT image analysis, and Grad-CAM visualization to
    support fast, intelligible screening for key macular conditions.
  </p>
  <div class="class-badges">
    <div class="class-badge">CNV</div>
    <div class="class-badge">DME</div>
    <div class="class-badge">DRUSEN</div>
    <div class="class-badge">NORMAL</div>
  </div>
</div>
"""

PROJECT_OVERVIEW_HTML = """
<div class="content-card">
  <div class="eyebrow">Project Overview</div>
  <h2 class="section-title">A clearer interface for an explainable OCT diagnosis workflow.</h2>
  <p class="body-copy">
    The platform presents the project like a professional healthcare AI product while keeping the
    core model pipeline intact. It explains the problem, showcases the workflow, and separates
    screening into a dedicated analysis workspace.
  </p>
</div>
"""

FEATURES_HTML = """
<div class="feature-grid">
  <div class="info-card"><div class="eyebrow">Purpose</div><h3>AI-assisted retinal screening</h3><p class="body-copy">The model classifies retinal OCT images into clinically relevant categories through a cleaner interface.</p></div>
  <div class="info-card"><div class="eyebrow">Explainability</div><h3>Visual reasoning support</h3><p class="body-copy">Grad-CAM heatmaps show where the network focused during inference, making the result more interpretable.</p></div>
  <div class="info-card"><div class="eyebrow">Model</div><h3>TensorFlow inference</h3><p class="body-copy">The backend loads a saved Keras model once and preprocesses each retinal image consistently before prediction.</p></div>
  <div class="info-card"><div class="eyebrow">Output</div><h3>Disease name focus</h3><p class="body-copy">The prediction page emphasizes the disease label, confidence level, and supporting visual outputs.</p></div>
</div>
"""

WORKFLOW_HTML = """
<div class="workflow-grid">
  <div class="workflow-card"><div class="eyebrow">01</div><h3>Image intake</h3><p class="body-copy">Upload a genuine retinal OCT image into the screening workspace.</p></div>
  <div class="workflow-card"><div class="eyebrow">02</div><h3>Model inference</h3><p class="body-copy">The app resizes and preprocesses the scan, then runs disease prediction using the trained Keras model.</p></div>
  <div class="workflow-card"><div class="eyebrow">03</div><h3>Explainable result</h3><p class="body-copy">Review the disease label, confidence meter, probability chart, original image, and Grad-CAM heatmap.</p></div>
</div>
"""


with gr.Blocks(css=CUSTOM_CSS, title="Retinal OCT Analysis", theme=gr.themes.Soft()) as demo:
    gr.HTML(HOME_HTML)

    with gr.Tabs():
        with gr.Tab("Home"):
            gr.HTML(PROJECT_OVERVIEW_HTML)
            gr.HTML(FEATURES_HTML)

        with gr.Tab("Project"):
            gr.HTML(
                """
                <div class="content-card">
                  <div class="eyebrow">Model Journey</div>
                  <h2 class="section-title">From uploaded image to interpretable output.</h2>
                  <p class="body-copy">
                    This Space adapts the original website into a single hosted ML interface while preserving the
                    project's emphasis on explainability, visual outputs, and professional presentation.
                  </p>
                </div>
                """
            )
            gr.HTML(WORKFLOW_HTML)

        with gr.Tab("Prediction"):
            gr.HTML(
                """
                <div class="warning-box">
                  <strong>Important Upload Warning:</strong> Please upload genuine retinal OCT images only.
                  Random images may still receive a class label, but that output is not medically meaningful.
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(
                        type="numpy",
                        label="Upload retinal OCT scan",
                        image_mode="RGB",
                    )
                    analyze_btn = gr.Button("Analyze Scan", variant="primary")
                with gr.Column(scale=6):
                    result_html = gr.HTML(
                        """
                        <div class="result-card">
                          <div class="eyebrow">Awaiting Input</div>
                          <h2>Results will appear here.</h2>
                          <p class="body-copy">Upload a retinal OCT image and run prediction to view the disease label, confidence level, and visual explanation outputs.</p>
                        </div>
                        """
                    )

            with gr.Row():
                original_output = gr.Image(label="Original Scan", type="pil")
                heatmap_output = gr.Image(label="Heatmap Output", type="pil")

            probability_plot = gr.Plot(label="Class Probability Plot")

            analyze_btn.click(
                fn=analyze_image,
                inputs=image_input,
                outputs=[result_html, original_output, heatmap_output, probability_plot],
            )


if __name__ == "__main__":
    demo.launch()
