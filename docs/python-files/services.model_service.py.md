# `services/model_service.py`

## Purpose

`services/model_service.py` is responsible for loading the trained Keras model and running prediction on an input retinal image.

## Main Responsibilities

- load the `.keras` model once at import time
- ensure the model file exists locally before loading
- download the model automatically when it is missing
- load class names from `class_names.json`
- preprocess a retinal image for ResNet50
- run inference
- return the predicted class and raw probabilities

## Global Objects

### `model`

Created using:

```python
tf.keras.models.load_model(MODEL_PATH)
```

This means the model is loaded only once when the module is first imported, which is more efficient than loading it inside every request.

Before loading, the module now calls `ensure_model_available()`. This gives the backend a dual behavior:

- local-first: use the existing local `oct_retinal_model.keras` file if present
- cloud fallback: download the model from the configured Google Drive URL if the file is missing

### `class_names`

Loaded from `class_names.json`. This converts the model's output index into a human-readable disease label.

## Model Availability Helpers

### `ensure_model_available()`

This function checks whether the model file already exists.

If the file exists:

- it returns the local path immediately

If the file does not exist:

- it checks whether automatic download is enabled
- downloads the model from the configured URL
- saves it to `MODEL_PATH`
- returns the final local path

### `_download_model_from_drive(download_url, destination)`

This helper performs the Google Drive download flow. It handles:

- the first request to the Drive URL
- confirmation token extraction for large file downloads
- the final streamed download to disk

This is important because large Google Drive files often require a confirmation step before the actual download begins.

## Main Function

### `predict_image(img_path)`

This function performs the full inference preparation and prediction process.

## Detailed Flow

1. Load the image from disk with target size `(224, 224)`.
2. Convert it into a numeric tensor.
3. Add a batch dimension using `tf.expand_dims`.
4. Apply `tf.keras.applications.resnet50.preprocess_input`.
5. Run `model.predict(...)`.
6. Select the index with the highest probability using `np.argmax`.
7. Return:
   - predicted class label
   - raw class probabilities
   - preprocessed image array

## Why The Returned `img_array` Matters

The prediction route reuses the returned preprocessed image tensor for Grad-CAM generation, so the image does not need to be reloaded and reprocessed separately.

## Related Files

- [services/model_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\model_service.py)
- [config.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\config.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)
- [services/gradcam_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\gradcam_service.py)
