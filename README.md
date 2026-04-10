# Retinal OCT Analysis

Retinal OCT Analysis is a deep learning web application for classifying retinal OCT scans into clinically relevant categories and visualizing model attention with Grad-CAM. The project combines a Flask backend, a React frontend, a trained Keras model, and supporting utilities for upload handling and result generation.

## What This Project Does

- accepts a retinal OCT image upload from the web UI
- runs disease classification using a trained `.keras` model
- returns the predicted class and confidence score
- generates a Grad-CAM heatmap to show model attention
- displays the original scan and explanation visuals in the frontend

## Supported Classes

- `CNV`
- `DME`
- `DRUSEN`
- `NORMAL`

## Tech Stack

- Backend: Flask, TensorFlow / Keras, NumPy, Matplotlib
- Frontend: React, Vite, Tailwind CSS
- Model format: `oct_retinal_model.keras`
- Model delivery: local file first, Google Drive auto-download fallback

## Project Structure

```text
Project_OCT/
|-- app.py
|-- config.py
|-- class_names.json
|-- oct_retinal_model.keras
|-- routes/
|-- services/
|-- utils/
|-- static/
|-- frontend/
|-- docs/
`-- README.md
```

## Backend Overview

The Flask backend exposes the prediction endpoint and coordinates the inference pipeline.

Main responsibilities:

- save uploaded images
- load the trained Keras model
- preprocess the image for inference
- compute prediction probabilities
- calculate confidence
- generate Grad-CAM overlays
- return JSON results for the frontend

Main files:

- `app.py`
- `routes/predict.py`
- `services/model_service.py`
- `services/gradcam_service.py`
- `services/plot_service.py`

## Frontend Overview

The React frontend provides a polished multi-page interface with:

- a homepage
- a project overview page
- a dedicated prediction page
- upload controls
- result cards for prediction, confidence, and heatmap output

## How The Workflow Runs

1. The user opens the prediction page.
2. A retinal OCT image is uploaded.
3. The backend stores the file in `static/uploads`.
4. The trained model predicts the disease class.
5. The system calculates the top confidence score.
6. A Grad-CAM heatmap and probability plot are generated.
7. The frontend displays the prediction, confidence bar, original image, and heatmap.

## How To Run Locally

Model behavior:

- if `oct_retinal_model.keras` already exists in the project root, the backend uses it directly
- if the model file is missing, the backend tries to download it automatically from the configured Google Drive link
- the first startup may take longer when the model needs to be downloaded

### Backend

From the project root:

```powershell
python app.py
```

Backend URL:

```text
http://127.0.0.1:5000
```

Optional environment variables:

```powershell
$env:MODEL_DOWNLOAD_URL="https://drive.google.com/uc?export=download&id=1jXrsLBiQHOnukSr1dg3CIeUXFshr05cF"
$env:AUTO_DOWNLOAD_MODEL="1"
```

### Frontend

From the `frontend` folder:

```powershell
npm install
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

## Current Highlights

- trained retinal OCT classifier integrated into a web app
- local-first model loading with cloud download fallback
- confidence level shown in the prediction UI
- Grad-CAM explanation output included with the result
- cleaned frontend branding and navigation
- per-file Python documentation available in `docs/python-files`

## Limitations

- the model is intended for retinal OCT scans only
- non-OCT images may still produce a label, but that result is not meaningful
- Grad-CAM provides useful visual explanation, but heatmaps can still be coarse

## What's Next

Planned or recommended next improvements:

- wire `ood_service.py` into the prediction pipeline for reliability warnings
- wire `heatmap_interpretation_service.py` into the API response for richer explanation text
- add backend requirements and setup instructions in a dedicated environment guide
- add frontend and backend automated tests
- improve model evaluation reporting with metrics such as accuracy, precision, recall, and confusion matrix
- compare Grad-CAM with stronger explainability methods for finer lesion localization
- prepare the app for deployment with environment-based configuration

## Render Deployment Note

The model file is intentionally kept out of GitHub because it exceeds GitHub's file size limit. In deployment:

- Render starts the backend
- if the model file is not present on disk, the backend downloads it automatically from Google Drive
- once downloaded, the backend loads it and serves predictions normally

This lets the same codebase work both locally and in cloud deployment without requiring the large model file to live in the Git repository.

## Documentation

Detailed Python module documentation is available here:

- `docs/python-files/README.md`

## Important Note

This project is for retinal OCT image analysis and research-oriented demonstration. Model outputs should be interpreted carefully and not treated as standalone clinical decisions.
