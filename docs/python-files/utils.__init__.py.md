# `utils/__init__.py`

## Purpose

`utils/__init__.py` is an intentionally empty package marker for the `utils` folder.

## Why It Exists

The `utils` package contains helper functions that support the main workflow. Keeping `__init__.py` here helps package imports remain explicit and stable.

## Example

The project imports `save_uploaded_file` like this:

```python
from utils.file_utils import save_uploaded_file
```

## Should It Remain Empty?

Yes. If no package-level initialization is needed, an empty `__init__.py` is completely normal.

## Related Files

- [utils/__init__.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\utils\__init__.py)
- [utils/file_utils.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\utils\file_utils.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)

