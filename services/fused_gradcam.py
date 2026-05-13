import os

import numpy as np
import tensorflow as tf


TARGET_IMAGE_SIZE = (224, 224)


def _get_base_model(model):
    return next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), model)


def _get_head_model(model, base_model):
    last_conv_shape = base_model.output.shape[1:]
    classifier_input = tf.keras.Input(shape=last_conv_shape)
    x = classifier_input
    base_model_idx = model.layers.index(base_model)

    for layer in model.layers[base_model_idx + 1:]:
        x = layer(x)

    return tf.keras.Model(classifier_input, x)


def _normalize_heatmap(heatmap):
    heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)
    maximum = float(np.max(heatmap))
    if maximum <= 0:
        return np.zeros_like(heatmap, dtype=np.float32)
    return heatmap / maximum


def _resize_heatmap(heatmap, target_size=TARGET_IMAGE_SIZE):
    resized = tf.image.resize(
        heatmap[..., np.newaxis],
        target_size,
        method="bilinear",
    ).numpy().squeeze()
    return _normalize_heatmap(resized)


def _resolve_multilayer_names(model):
    base_model = _get_base_model(model)
    base_layer_names = [layer.name for layer in base_model.layers]
    desired_names = {
        "layer2": "conv3_block4_out",
        "layer3": "conv4_block6_out",
        "layer4": "conv5_block3_out",
    }

    if all(name in base_layer_names for name in desired_names.values()):
        return desired_names

    base_model.summary()
    conv_out_layers = [
        layer.name for layer in base_model.layers
        if "conv" in layer.name and layer.name.endswith("_out")
    ]
    if not conv_out_layers:
        raise ValueError("Could not find convolutional residual output layers in the loaded model.")

    fallback_indices = [
        max(0, len(conv_out_layers) // 3 - 1),
        max(0, (2 * len(conv_out_layers)) // 3 - 1),
        len(conv_out_layers) - 1,
    ]
    return {
        "layer2": conv_out_layers[fallback_indices[0]],
        "layer3": conv_out_layers[fallback_indices[1]],
        "layer4": conv_out_layers[fallback_indices[2]],
    }


def _substitute_dead_cams(cams):
    order = ["cam2", "cam3", "cam4"]

    def first_nonzero(candidates):
        for name in candidates:
            candidate = cams[name]
            if float(np.max(candidate)) > 0:
                return candidate.copy()
        return None

    if float(np.max(cams["cam4"])) <= 0:
        replacement = first_nonzero(["cam3", "cam2"])
        if replacement is not None:
            cams["cam4"] = replacement

    if float(np.max(cams["cam3"])) <= 0:
        replacement = first_nonzero(["cam4", "cam2"])
        if replacement is not None:
            cams["cam3"] = replacement

    if float(np.max(cams["cam2"])) <= 0:
        replacement = first_nonzero(["cam3", "cam4"])
        if replacement is not None:
            cams["cam2"] = replacement

    for name in order:
        cams[name] = _normalize_heatmap(cams[name])

    return cams


def get_gradcam_single_layer(model, img_array, layer_name, class_idx):
    """
    Standard Grad-CAM for one layer.
    Uses GradientTape to get gradients of class score w.r.t. that layer's
    feature maps. Returns normalized heatmap upsampled to 224x224.
    """
    base_model = _get_base_model(model)
    target_layer = base_model.get_layer(layer_name)
    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[target_layer.output, base_model.output],
    )
    head_model = _get_head_model(model, base_model)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = feature_extractor(img_array)
        tape.watch(conv_outputs)
        preds = head_model(base_output)
        target_score = preds[:, class_idx]

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    if float(tf.reduce_max(heatmap).numpy()) <= 0:
        return np.zeros(TARGET_IMAGE_SIZE, dtype=np.float32)

    return _resize_heatmap(heatmap.numpy())


def get_multilayer_gradcam(model, img_array, class_idx):
    """
    Run Grad-CAM on Layer2, Layer3, and Layer4 of ResNet50.

    For ResNet50 in Keras, the layer names to use are:
      - Layer2 equivalent: 'conv3_block4_out'   (28x28 feature maps)
      - Layer3 equivalent: 'conv4_block6_out'   (14x14 feature maps)
      - Layer4 equivalent: 'conv5_block3_out'   (7x7 feature maps)

    If any of those layer names don't exist in the loaded model,
    print model.summary() to find the correct names and use the
    last layer of each residual block group instead.

    Returns dict: {'cam2': array, 'cam3': array, 'cam4': array}
    All normalized to [0,1] and upsampled to 224x224.
    """
    layer_names = _resolve_multilayer_names(model)
    cams = {
        "cam2": get_gradcam_single_layer(model, img_array, layer_names["layer2"], class_idx),
        "cam3": get_gradcam_single_layer(model, img_array, layer_names["layer3"], class_idx),
        "cam4": get_gradcam_single_layer(model, img_array, layer_names["layer4"], class_idx),
    }
    return _substitute_dead_cams(cams)


def fuse_cams(cam2, cam3, cam4, alpha2=0.2, alpha3=0.3, alpha4=0.5):
    """
    Weighted combination: H_fusion = alpha2*H2 + alpha3*H3 + alpha4*H4
    Normalize the result to [0,1].
    Returns fused heatmap as numpy array (224x224).
    """
    fused = alpha2 * cam2 + alpha3 * cam3 + alpha4 * cam4
    return _normalize_heatmap(fused)


def compute_adaptive_fusion_weights(layer_validation_scores, fallback_weights=None):
    if fallback_weights is None:
        fallback_weights = {
            "layer2": 0.2,
            "layer3": 0.3,
            "layer4": 0.5,
        }

    score_values = {
        "layer2": max(0.0, float(layer_validation_scores.get("layer2_mask_confidence", 0.0))),
        "layer3": max(0.0, float(layer_validation_scores.get("layer3_mask_confidence", 0.0))),
        "layer4": max(0.0, float(layer_validation_scores.get("layer4_mask_confidence", 0.0))),
    }

    score_sum = sum(score_values.values())
    if score_sum <= 0:
        return fallback_weights.copy()

    return {
        key: value / score_sum
        for key, value in score_values.items()
    }


def generate_binary_mask(fused_cam, threshold=0.5):
    """
    Threshold fused heatmap to binary mask.
    Returns float32 array with 0.0 or 1.0 values, shape (224,224).
    """
    return (np.asarray(fused_cam, dtype=np.float32) >= threshold).astype(np.float32)


def apply_mask_and_repredict(model, original_img_array, mask, preprocess_fn, class_idx):
    """
    Apply binary mask to original image (elementwise multiply).
    Run masked image through model.
    Return confidence score for class_idx.
    Masked image = original_img * mask[:,:,np.newaxis]
    """
    masked_img = np.asarray(original_img_array, dtype=np.float32) * mask[:, :, np.newaxis]
    batched = tf.expand_dims(masked_img, axis=0)
    processed = preprocess_fn(batched)
    preds = model.predict(processed, verbose=0)[0]
    return float(preds[class_idx])


def _load_raw_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=TARGET_IMAGE_SIZE)
    return tf.keras.utils.img_to_array(img).astype(np.float32)


def run_full_pipeline(model, img_path, class_labels, preprocess_fn):
    """
    Master function. Takes model + image path. Returns a dict with:
    {
      'original_img': numpy array (224,224,3) in [0,255] uint8,
      'standard_gradcam': heatmap from conv5_block3_out only (224,224),
      'cam2': heatmap from layer2 (224,224),
      'cam3': heatmap from layer3 (224,224),
      'cam4': heatmap from layer4 (224,224),
      'fused_cam': fused heatmap (224,224),
      'binary_mask': binary mask (224,224),
      'masked_img': masked OCT image (224,224,3),
      'predicted_class': string e.g. "CNV",
      'predicted_class_idx': int,
      'original_confidence': float e.g. 0.98,
      'validation': {
          'layer2_mask_confidence': float,
          'layer3_mask_confidence': float,
          'layer4_mask_confidence': float,
          'fused_mask_confidence': float,
      }
    }
    """
    original_img = _load_raw_image(img_path)
    processed = preprocess_fn(tf.expand_dims(original_img, axis=0))
    predictions = model.predict(processed, verbose=0)[0]
    predicted_class_idx = int(np.argmax(predictions))
    predicted_class = class_labels[predicted_class_idx]
    original_confidence = float(predictions[predicted_class_idx])

    standard_gradcam = get_gradcam_single_layer(model, processed, "conv5_block3_out", predicted_class_idx)
    multilayer = get_multilayer_gradcam(model, processed, predicted_class_idx)

    layer_validation = {
        "layer2_mask_confidence": apply_mask_and_repredict(
            model, original_img, generate_binary_mask(multilayer["cam2"]), preprocess_fn, predicted_class_idx
        ),
        "layer3_mask_confidence": apply_mask_and_repredict(
            model, original_img, generate_binary_mask(multilayer["cam3"]), preprocess_fn, predicted_class_idx
        ),
        "layer4_mask_confidence": apply_mask_and_repredict(
            model, original_img, generate_binary_mask(multilayer["cam4"]), preprocess_fn, predicted_class_idx
        ),
    }

    adaptive_weights = compute_adaptive_fusion_weights(layer_validation)
    fused_cam = fuse_cams(
        multilayer["cam2"],
        multilayer["cam3"],
        multilayer["cam4"],
        alpha2=adaptive_weights["layer2"],
        alpha3=adaptive_weights["layer3"],
        alpha4=adaptive_weights["layer4"],
    )
    binary_mask = generate_binary_mask(fused_cam)
    masked_img = np.clip(original_img * binary_mask[:, :, np.newaxis], 0, 255).astype(np.uint8)

    validation = {
        **layer_validation,
        "fused_mask_confidence": apply_mask_and_repredict(
            model, original_img, binary_mask, preprocess_fn, predicted_class_idx
        ),
    }

    dominant_layer = max(adaptive_weights, key=adaptive_weights.get)

    return {
        "original_img": np.clip(original_img, 0, 255).astype(np.uint8),
        "standard_gradcam": standard_gradcam,
        "cam2": multilayer["cam2"],
        "cam3": multilayer["cam3"],
        "cam4": multilayer["cam4"],
        "fused_cam": fused_cam,
        "binary_mask": binary_mask.astype(np.float32),
        "masked_img": masked_img,
        "predicted_class": predicted_class,
        "predicted_class_idx": predicted_class_idx,
        "original_confidence": original_confidence,
        "validation": validation,
        "adaptive_weights": adaptive_weights,
        "adaptive_strategy": {
            "method": "confidence-retention adaptive fusion",
            "dominant_layer": dominant_layer,
            "explanation": (
                "Layer weights are assigned per scan by normalizing the confidence retained by each "
                "layer-specific mask. Layers that preserve more predictive evidence contribute more "
                "strongly to the fused explanation map."
            ),
        },
        "layer_names": _resolve_multilayer_names(model),
        "image_path": img_path,
    }
