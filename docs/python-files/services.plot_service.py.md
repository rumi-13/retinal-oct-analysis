# `services/plot_service.py`

## Purpose

`services/plot_service.py` generates and saves a simple bar chart of class probabilities after a prediction is made.

## Main Function

### `save_probability_plot(class_names, probabilities, save_path)`

This function creates a Matplotlib figure showing the model's confidence for each class.

## Detailed Behavior

1. Create a figure with size `(5, 4)`.
2. Draw a bar chart using `class_names` as labels and `probabilities` as bar heights.
3. Set the plot title to `Class Probabilities`.
4. Label the y-axis as `Confidence`.
5. Force the y-axis range to `0` through `1`.
6. Save the plot to the provided path.
7. Close the figure so memory is released.

## Why This File Exists Separately

Separating plotting logic from route logic keeps `routes/predict.py` focused on request handling rather than visualization details.

## Output

The output is an image file written to disk, typically inside `static/uploads`, so the frontend can access it through a generated URL.

## Related Files

- [services/plot_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\plot_service.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)

