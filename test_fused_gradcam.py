import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from config import CLASS_NAMES_PATH, MODEL_PATH, OUTPUT_FOLDER
from services.fused_gradcam import run_full_pipeline
from services.visualize_results import save_visualization_results


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_fused_gradcam.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    class_labels = ["CNV", "DME", "DRUSEN", "NORMAL"]
    if os.path.exists(CLASS_NAMES_PATH):
        import json

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file_obj:
            class_labels = json.load(file_obj)

    results = run_full_pipeline(
        model=model,
        img_path=image_path,
        class_labels=class_labels,
        preprocess_fn=tf.keras.applications.resnet50.preprocess_input,
    )
    saved_paths = save_visualization_results(results, output_dir=OUTPUT_FOLDER, close_figure=False)

    print(f"Predicted class: {results['predicted_class']}")
    print(f"Original confidence: {results['original_confidence']:.4f}")
    print(f"Layer2 mask confidence: {results['validation']['layer2_mask_confidence']:.4f}")
    print(f"Layer3 mask confidence: {results['validation']['layer3_mask_confidence']:.4f}")
    print(f"Layer4 mask confidence: {results['validation']['layer4_mask_confidence']:.4f}")
    print(f"Fused mask confidence: {results['validation']['fused_mask_confidence']:.4f}")
    print(f"Result image saved to: {saved_paths['latest_result_path']}")

    plt.show()


if __name__ == "__main__":
    main()
