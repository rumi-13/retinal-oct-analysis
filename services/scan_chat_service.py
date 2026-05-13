import json
from urllib import error, request

from config import GEMINI_API_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL


def _build_scan_context(analysis):
    if not analysis:
        return "No scan analysis is currently available."

    predicted_class = analysis.get("predicted_class") or analysis.get("prediction", "Unknown")
    original_confidence = analysis.get("original_confidence")
    validation = analysis.get("validation", {})

    confidence_text = (
        f"{original_confidence * 100:.2f}%"
        if isinstance(original_confidence, (int, float))
        else "Unavailable"
    )

    validation_lines = []
    for key, value in validation.items():
        if isinstance(value, (int, float)):
            validation_lines.append(f"- {key}: {value * 100:.2f}% confidence retention")
        else:
            validation_lines.append(f"- {key}: {value}")

    validation_summary = "\n".join(validation_lines) if validation_lines else "No validation metrics provided."

    return (
        "Current retinal OCT fused analysis context:\n"
        f"- Predicted class: {predicted_class}\n"
        f"- Original confidence: {confidence_text}\n"
        f"- Validation metrics:\n{validation_summary}\n"
        "- Available visual outputs: uploaded scan, standard Grad-CAM, layer 2/3/4 heatmaps, fused CAM, fused heatmap, binary mask, masked OCT, combined validation figure.\n"
        "- This assistant must discuss the uploaded scan and these outputs only.\n"
        "- It must not provide a medical diagnosis or treatment advice.\n"
    )


def ask_scan_chat(question, analysis):
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        raise RuntimeError(
            "Scan chatbot is not configured. Set GEMINI_API_KEY and GEMINI_MODEL in your environment or .env.local."
        )

    system_prompt = (
        "You are a specialized assistant for retinal OCT scan result review. "
        "Answer only questions about the current uploaded scan and its generated outputs. "
        "Explain Grad-CAM, fused CAM, masks, and validation metrics in clear academic language. "
        "Do not claim to be a doctor. Do not provide treatment advice. "
        "If the question goes beyond the scan, redirect back to the current analysis."
    )

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": system_prompt,
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{_build_scan_context(analysis)}\n\nUser question: {question}",
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    req = request.Request(
        url=f"{GEMINI_API_BASE_URL}/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Chat provider rejected the request: {details or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Chat provider request failed: {exc.reason}") from exc

    candidates = response_payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    content = candidates[0].get("content", {})
    parts = content.get("parts") or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    answer = "\n".join(text_parts).strip()

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    return answer
