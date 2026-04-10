# `config.py`

## Purpose

`config.py` centralizes the backend configuration values used by the project. Its main job is to build stable absolute paths for the model, class names file, upload folder, and base API URL.

## Variables Defined

- `BASE_DIR`
  The absolute path to the project root where `config.py` lives.
- `MODEL_PATH`
  The full path to `oct_retinal_model.keras`.
- `CLASS_NAMES_PATH`
  The full path to `class_names.json`.
- `MODEL_FILE_ID`
  The Google Drive file identifier for the model artifact.
- `MODEL_DOWNLOAD_URL`
  The model download URL. It can be overridden with the `MODEL_DOWNLOAD_URL` environment variable.
- `AUTO_DOWNLOAD_MODEL`
  A boolean-style flag that controls whether the backend should try to download the model automatically if it is missing locally.
- `UPLOAD_FOLDER`
  The full path to `static/uploads`, where uploaded scans and generated visual artifacts are saved.
- `BASE_URL`
  The backend base URL currently set to `http://127.0.0.1:5000/`.

## Why Paths Are Built This Way

Using `os.path.join(BASE_DIR, ...)` avoids hardcoding full machine-specific paths. That makes the project more portable and reduces path mistakes when running the app from different working directories.

## Where It Is Used

- `app.py`
  Uses `UPLOAD_FOLDER` to make sure the upload directory exists at startup.
- `services/model_service.py`
  Uses `MODEL_PATH` and `CLASS_NAMES_PATH` to load the trained model and labels, and uses the model download configuration when the file is missing.
- `routes/predict.py`
  Uses `UPLOAD_FOLDER` and `BASE_URL` to save artifacts and generate returned URLs.
- `utils/file_utils.py`
  Uses `UPLOAD_FOLDER` when saving uploaded files.

## Design Role

This file keeps configuration separate from business logic. Instead of spreading model paths and folder paths throughout the codebase, the rest of the application imports them from one place.

## Related Files

- [config.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\config.py)
- [services/model_service.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\services\model_service.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)
