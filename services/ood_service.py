import numpy as np


def assess_ood(probabilities, class_names, confidence_threshold=0.55, margin_threshold=0.20, entropy_threshold=0.90):
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / np.sum(probs)

    sorted_indices = np.argsort(probs)[::-1]
    top_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_index

    top_confidence = float(probs[top_index])
    second_confidence = float(probs[second_index])
    confidence_margin = top_confidence - second_confidence

    entropy = float(-np.sum(probs * np.log(probs)))
    normalized_entropy = entropy / np.log(len(probs)) if len(probs) > 1 else 0.0

    is_ood = bool(
        top_confidence < confidence_threshold
        or confidence_margin < margin_threshold
        or normalized_entropy > entropy_threshold
    )

    reasons = []
    if top_confidence < confidence_threshold:
      reasons.append("top confidence is lower than the expected in-distribution range")
    if confidence_margin < margin_threshold:
      reasons.append("the model is not clearly separating the best class from the next closest class")
    if normalized_entropy > entropy_threshold:
      reasons.append("the output probabilities are too diffuse across classes")

    if is_ood:
        title = "Possible out-of-distribution input"
        summary = (
            "This scan may be outside the model's familiar training distribution, or it may be too ambiguous "
            "for a reliable class decision."
        )
        recommendation = (
            "Review the image manually and treat the predicted label as provisional rather than definitive."
        )
    else:
        title = "Input appears in distribution"
        summary = (
            f"The probability pattern is reasonably consistent with the trained classes, with {class_names[top_index]} "
            "remaining the leading prediction."
        )
        recommendation = "Proceed with the model output alongside normal clinical review."

    return {
        "is_ood": is_ood,
        "title": title,
        "summary": summary,
        "recommendation": recommendation,
        "metrics": {
            "top_confidence": float(round(top_confidence, 4)),
            "second_confidence": float(round(second_confidence, 4)),
            "confidence_margin": float(round(confidence_margin, 4)),
            "normalized_entropy": float(round(normalized_entropy, 4)),
        },
        "reasons": reasons,
    }
