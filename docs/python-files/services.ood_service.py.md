# `services/ood_service.py`

## Purpose

`services/ood_service.py` estimates whether a prediction may be out-of-distribution or unreliable based on the output probability pattern.

## What Out-Of-Distribution Means Here

In this project, out-of-distribution does not prove that an image is completely unrelated to OCT. Instead, it is a heuristic signal that the prediction may be uncertain, ambiguous, or unlike the examples the model learned during training.

## Main Function

### `assess_ood(probabilities, class_names, confidence_threshold=0.55, margin_threshold=0.20, entropy_threshold=0.90)`

This function takes the probability vector from the model and evaluates its quality.

## Metrics Computed

### Top confidence

The highest predicted probability.

### Second confidence

The second-highest predicted probability.

### Confidence margin

The difference between the top and second class probabilities.

### Normalized entropy

A measure of how spread out the distribution is. High entropy means the model is uncertain across many classes.

## Decision Rule

The result is considered potentially OOD if any of the following are true:

- top confidence is too low
- confidence margin is too small
- normalized entropy is too high

## Output

The function returns a dictionary containing:

- `is_ood`
- `title`
- `summary`
- `recommendation`
- `metrics`
- `reasons`

This structure is ready for API or frontend use.

## Current Status In The Project

This file contains useful logic but is not currently invoked from `routes/predict.py`. That means the capability exists but is not presently active in the response pipeline.

## Related Files

- [services/ood_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\ood_service.py)
- [services/model_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\model_service.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)

