# `routes/predict.py`

## Purpose

`routes/predict.py` contains the Flask API endpoint that handles retinal image prediction. It is the main request-processing layer of the backend.

## Core Responsibilities

- accept uploaded image files from the frontend
- validate the request
- save the uploaded image
- run model inference
- calculate prediction confidence
- generate a probability plot
- generate a Grad-CAM heatmap image
- return the prediction result as JSON

## Main Objects And Functions

### `predict_bp`

A Flask `Blueprint` named `"predict"`. This lets the prediction route be defined separately and later registered in `app.py`.

### `save_gradcam_image(img_path, heatmap, output_path, alpha=0.4)`

This helper function overlays the Grad-CAM heatmap on top of the original retinal image.

What it does:

- loads the original image
- converts the heatmap values to `0-255`
- colors the heatmap using the `jet` colormap
- resizes it to the image size
- blends the heatmap with the original image
- saves the combined visualization

### `predict()`

This is the Flask view function behind `POST /predict`.

## Request Handling Flow

1. Check whether the uploaded file exists in `request.files`.
2. Check whether the filename is non-empty.
3. Save the uploaded file using `utils.file_utils.save_uploaded_file`.
4. Call `predict_image(filepath)` from `services.model_service`.
5. Receive:
   - predicted class label
   - class probability vector
   - preprocessed image tensor
6. Compute the top confidence score.
7. Save a class probability bar chart using `services.plot_service`.
8. Generate a Grad-CAM heatmap using `services.gradcam_service`.
9. If heatmap generation succeeds, save the overlay image.
10. Return a JSON response to the frontend.

## Response Fields

- `prediction`
  The predicted disease class name.
- `confidence_score`
  Numeric confidence value between `0` and `1`.
- `confidence`
  Human-readable percentage string, such as `92.14%`.
- `image_url`
  URL to the original uploaded image.
- `plot_url`
  URL to the generated probability plot.
- `cam_url`
  URL to the generated Grad-CAM overlay image, or `null` if heatmap creation fails.

## Design Notes

- This file is intentionally thin in some areas and delegates specialized work to services.
- Model loading is not done here, which avoids repeatedly loading the network for every request.
- The file focuses on orchestration: request in, workflow execution, response out.

## Current Observations

- `services.ood_service.py` and `services.heatmap_interpretation_service.py` exist but are not currently used here.
- The route currently supports one uploaded file per request.

## Related Files

- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)
- [services/model_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\model_service.py)
- [services/gradcam_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\gradcam_service.py)
- [services/plot_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\plot_service.py)
- [utils/file_utils.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\utils\file_utils.py)

