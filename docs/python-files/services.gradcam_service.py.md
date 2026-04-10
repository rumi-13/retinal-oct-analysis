# `services/gradcam_service.py`

## Purpose

`services/gradcam_service.py` generates the Grad-CAM attention heatmap for a preprocessed retinal image tensor. This is the explainability layer of the backend.

## Main Responsibility

It identifies which spatial regions in the image contributed most strongly to the model's predicted class.

## Imported Dependencies

- `tensorflow as tf`
  Used to inspect the model and compute gradients.
- `numpy as np`
  Used for array conversion and output normalization.
- `matplotlib`
  Configured with the `'Agg'` backend to avoid GUI requirements on the server.
- `model` from `services.model_service`
  Reuses the already-loaded Keras model.

## Main Function

### `get_gradcam_heatmap(img_array, last_conv_layer_name="conv5_block3_out")`

This function computes the Grad-CAM map.

## Step-By-Step Logic

1. Find the nested base CNN model inside the full classifier.
2. Locate the last convolutional layer by name.
3. Build `conv_model` to output the convolutional feature maps.
4. Build `classifier_model` from the layers that follow the base model.
5. Use `tf.GradientTape()` to compute gradients of the top predicted class with respect to the convolutional outputs.
6. Average gradients across spatial dimensions to get channel importance.
7. Weight the convolutional feature maps by those averaged gradients.
8. Collapse the channels into a single 2D heatmap.
9. Apply ReLU-like filtering with `tf.maximum(heatmap, 0)`.
10. Normalize the heatmap and return it as a NumPy array.

## Inputs

- `img_array`
  The already preprocessed image tensor, typically shaped like `(1, 224, 224, 3)`.
- `last_conv_layer_name`
  The convolutional layer to use for Grad-CAM. The default is `conv5_block3_out`, which matches ResNet50.

## Output

- Returns a normalized 2D NumPy heatmap.
- Returns `None` if the base model cannot be found.

## Why This File Is Important

This file makes the prediction explainable rather than opaque. The frontend uses its output to show which parts of the scan influenced the model.

## Related Files

- [services/gradcam_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\gradcam_service.py)
- [services/model_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\model_service.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)

