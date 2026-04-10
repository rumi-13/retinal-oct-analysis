# `app.py`

## Purpose

`app.py` is the application entry point for the Flask backend. It creates the Flask app instance, enables CORS, ensures the upload folder exists, registers the prediction routes, and starts the server when the file is run directly.

## Main Responsibilities

- create the Flask application object
- allow cross-origin requests from the frontend
- prepare the upload directory on startup
- attach the prediction blueprint
- run the development server on port `5000`

## Important Imports

- `Flask` from `flask`
  Creates the backend web application.
- `CORS` from `flask_cors`
  Allows the React frontend to call the Flask API from another origin.
- `predict_bp` from `routes.predict`
  The blueprint that contains the `/predict` endpoint.
- `UPLOAD_FOLDER` from `config`
  The configured location where uploaded images and generated outputs are stored.

## Execution Flow

1. A Flask app is created with `static_folder='static'`.
2. CORS is enabled for the whole app.
3. `os.makedirs(UPLOAD_FOLDER, exist_ok=True)` ensures the upload directory exists before any request tries to save a file.
4. The prediction blueprint is registered.
5. If the file is run directly, Flask starts in debug mode on port `5000`.

## Why This File Matters

This file is the glue that turns the backend modules into a working web server. Without it, the route logic in `routes/predict.py` would exist, but nothing would expose it as a running API.

## Related Files

- [app.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\app.py)
- [routes/predict.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\routes\predict.py)
- [config.py](C:\Users\Rumi\Desktop\RESEARCH\Dataset_others\Project_OCT\config.py)

