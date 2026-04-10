import numpy as np


def _describe_vertical_band(y_position):
    if y_position < 0.33:
        return "upper"
    if y_position < 0.66:
        return "central"
    return "lower"


def _describe_horizontal_band(x_position):
    if x_position < 0.33:
        return "left"
    if x_position < 0.66:
        return "central"
    return "right"


def generate_heatmap_interpretation(heatmap, prediction):
    if heatmap is None:
        return {
            "title": "Heatmap unavailable",
            "summary": "The Grad-CAM map could not be generated for this image, so no attention interpretation is available.",
            "clinical_note": "Interpretation requires a valid heatmap.",
        }

    normalized = np.nan_to_num(heatmap.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = np.max(normalized)

    if maximum <= 0:
        return {
            "title": "Low activation pattern",
            "summary": "The model did not produce a strong localized attention area in this scan.",
            "clinical_note": "Treat the prediction with caution because the visual explanation is diffuse.",
        }

    normalized = normalized / maximum
    threshold = 0.6
    hot_mask = normalized >= threshold

    if not np.any(hot_mask):
        hot_mask = normalized >= 0.4

    coordinates = np.argwhere(hot_mask)

    if coordinates.size == 0:
        y_center = 0.5
        x_center = 0.5
    else:
        y_center = float(np.mean(coordinates[:, 0]) / normalized.shape[0])
        x_center = float(np.mean(coordinates[:, 1]) / normalized.shape[1])

    vertical_band = _describe_vertical_band(y_center)
    horizontal_band = _describe_horizontal_band(x_center)

    hot_ratio = float(np.mean(hot_mask))
    mean_activation = float(np.mean(normalized[hot_mask])) if np.any(hot_mask) else 0.0

    if hot_ratio < 0.08:
        spread_description = "a compact attention region"
    elif hot_ratio < 0.2:
        spread_description = "a moderately focused attention region"
    else:
        spread_description = "a broad attention region"

    if mean_activation > 0.82:
        strength_description = "strong"
    elif mean_activation > 0.65:
        strength_description = "clear"
    else:
        strength_description = "mild"

    return {
        "title": f"Attention centered in the {vertical_band}-{horizontal_band} retinal area",
        "summary": (
            f"For the predicted class {prediction}, the Grad-CAM map shows {spread_description} "
            f"with {strength_description} activation around the {vertical_band}-{horizontal_band} portion of the scan."
        ),
        "clinical_note": (
            "This text explains where the model focused visually. It is an AI attention summary, "
            "not a medical conclusion, and should be interpreted alongside clinical review."
        ),
    }
