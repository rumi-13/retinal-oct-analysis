# `services/__init__.py`

## Purpose

`services/__init__.py` is an intentionally empty package marker for the `services` directory.

## Why It Matters

The `services` folder contains the reusable business logic of the backend, such as model inference, Grad-CAM generation, plotting, and optional reliability analysis. This file helps the folder behave as a Python package and keeps imports straightforward.

## Example Import Pattern

```python
from services.model_service import predict_image
```

## Should It Stay Empty?

Yes, it can remain empty if there is no package-level setup needed. Empty `__init__.py` files are normal and useful.

## Related Files

- [services/__init__.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\__init__.py)
- [services/model_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\model_service.py)
- [services/gradcam_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\gradcam_service.py)

