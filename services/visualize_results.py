import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def _heatmap_overlay(original_img, heatmap):
    original_img = np.asarray(original_img, dtype=np.float32)
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    jet = matplotlib.colormaps.get_cmap("jet")
    colored = jet(heatmap_uint8)[:, :, :3] * 255.0
    blended = 0.55 * original_img + 0.45 * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_figure(results):
    original_confidence = results["original_confidence"]
    fused_confidence = results["validation"]["fused_mask_confidence"]
    adaptive_weights = results.get("adaptive_weights", {})
    weight_text = (
        f"L2={adaptive_weights.get('layer2', 0):.2f}, "
        f"L3={adaptive_weights.get('layer3', 0):.2f}, "
        f"L4={adaptive_weights.get('layer4', 0):.2f}"
    )

    standard_overlay = _heatmap_overlay(results["original_img"], results["standard_gradcam"])
    fused_overlay = _heatmap_overlay(results["original_img"], results["fused_cam"])

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=[2.2, 1.1], hspace=0.25, wspace=0.12)

    panels = [
        ("Original OCT", results["original_img"], None),
        ("Standard Grad-CAM overlay", standard_overlay, None),
        ("Adaptive Fused CAM overlay", fused_overlay, None),
        ("Masked OCT", results["masked_img"], None),
    ]

    for idx, (title, image, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[0, idx])
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    ax_bar = fig.add_subplot(grid[1, :])
    bar_labels = ["Layer2 mask", "Layer3 mask", "Layer4 mask", "Fused mask", "Original (no mask)"]
    bar_values = [
        results["validation"]["layer2_mask_confidence"],
        results["validation"]["layer3_mask_confidence"],
        results["validation"]["layer4_mask_confidence"],
        results["validation"]["fused_mask_confidence"],
        original_confidence,
    ]
    colors = ["green", "yellow", "red", "purple", "gray"]
    y_positions = np.arange(len(bar_labels))

    bars = ax_bar.barh(y_positions, bar_values, color=colors, alpha=0.82)
    ax_bar.set_yticks(y_positions, bar_labels)
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_xlabel("Confidence")
    ax_bar.set_title("Confidence Retention Validation", fontsize=14, fontweight="bold")
    ax_bar.axvline(original_confidence, linestyle="--", color="black", linewidth=1.4)
    ax_bar.invert_yaxis()

    for bar, value in zip(bars, bar_values):
        ax_bar.text(
            min(value + 0.01, 0.985),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"Adaptive Fused Grad-CAM - {results['predicted_class']} ({original_confidence:.2f}) | "
        f"Fused retention: {fused_confidence:.2f} | Weights: {weight_text}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)
    return fig, standard_overlay, fused_overlay


def save_visualization_results(results, output_dir=None, close_figure=True):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")

    os.makedirs(output_dir, exist_ok=True)

    fig, standard_overlay, fused_overlay = _build_figure(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamped_result_path = os.path.join(output_dir, f"gradlrp_result_{timestamp}.png")
    latest_result_path = os.path.join(output_dir, "gradlrp_result_latest.png")
    latest_standard_overlay_path = os.path.join(output_dir, "standard_gradcam_latest.png")
    latest_fused_overlay_path = os.path.join(output_dir, "fused_cam_latest.png")
    latest_layer2_heatmap_path = os.path.join(output_dir, "layer2_heatmap_latest.png")
    latest_layer3_heatmap_path = os.path.join(output_dir, "layer3_heatmap_latest.png")
    latest_layer4_heatmap_path = os.path.join(output_dir, "layer4_heatmap_latest.png")
    latest_fused_heatmap_path = os.path.join(output_dir, "fused_heatmap_latest.png")
    latest_binary_mask_path = os.path.join(output_dir, "binary_mask_latest.png")
    latest_masked_oct_path = os.path.join(output_dir, "masked_oct_latest.png")

    fig.savefig(timestamped_result_path, dpi=180, bbox_inches="tight")
    fig.savefig(latest_result_path, dpi=180, bbox_inches="tight")
    plt.imsave(latest_standard_overlay_path, standard_overlay)
    plt.imsave(latest_fused_overlay_path, fused_overlay)
    plt.imsave(latest_layer2_heatmap_path, results["cam2"], cmap="jet")
    plt.imsave(latest_layer3_heatmap_path, results["cam3"], cmap="jet")
    plt.imsave(latest_layer4_heatmap_path, results["cam4"], cmap="jet")
    plt.imsave(latest_fused_heatmap_path, results["fused_cam"], cmap="jet")
    plt.imsave(latest_binary_mask_path, results["binary_mask"], cmap="gray")
    plt.imsave(latest_masked_oct_path, results["masked_img"])

    if close_figure:
        plt.close(fig)

    return {
        "figure": fig,
        "timestamped_result_path": timestamped_result_path,
        "latest_result_path": latest_result_path,
        "latest_standard_overlay_path": latest_standard_overlay_path,
        "latest_fused_overlay_path": latest_fused_overlay_path,
        "latest_layer2_heatmap_path": latest_layer2_heatmap_path,
        "latest_layer3_heatmap_path": latest_layer3_heatmap_path,
        "latest_layer4_heatmap_path": latest_layer4_heatmap_path,
        "latest_fused_heatmap_path": latest_fused_heatmap_path,
        "latest_binary_mask_path": latest_binary_mask_path,
        "latest_masked_oct_path": latest_masked_oct_path,
    }
