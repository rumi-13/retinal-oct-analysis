# `services/heatmap_interpretation_service.py`

## Purpose

`services/heatmap_interpretation_service.py` converts a raw Grad-CAM heatmap into short human-readable descriptive text.

## Main Idea

Instead of only showing the heatmap image, this file tries to summarize:

- where the strongest activation is located
- how concentrated or spread out the activation is
- how strong the activation appears

## Helper Functions

### `_describe_vertical_band(y_position)`

Maps a normalized vertical position to:

- `upper`
- `central`
- `lower`

### `_describe_horizontal_band(x_position)`

Maps a normalized horizontal position to:

- `left`
- `central`
- `right`

These helpers transform coordinate values into readable spatial terms.

## Main Function

### `generate_heatmap_interpretation(heatmap, prediction)`

This function returns a structured dictionary describing the attention pattern.

## Detailed Flow

1. If `heatmap` is `None`, return a fallback explanation saying no interpretation is available.
2. Replace invalid numeric values using `np.nan_to_num`.
3. Check whether there is any meaningful activation.
4. Normalize the heatmap by its maximum value.
5. Create a hot region mask using a threshold of `0.6`.
6. If that is too strict, lower the threshold to `0.4`.
7. Find the coordinates of the active region.
8. Compute the average center of activation.
9. Convert the center coordinates into verbal location labels.
10. Measure the active region size and average activation strength.
11. Produce descriptive language such as:
    - compact attention region
    - moderately focused attention region
    - broad attention region
12. Return a dictionary with:
    - `title`
    - `summary`
    - `clinical_note`

## Output Structure

The return value is designed for frontend display and includes short readable text rather than raw numeric arrays.

## Current Status In The Project

This file contains meaningful feature logic, but it is not currently called from `routes/predict.py`. That means the interpretation capability exists in code, but it is not yet active in the API response path.

## Related Files

- [services/heatmap_interpretation_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\heatmap_interpretation_service.py)
- [services/gradcam_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\gradcam_service.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)

