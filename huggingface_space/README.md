# Hugging Face Space Files

This folder contains a Space-ready Gradio version of the Retinal OCT Analysis project.

Files to place in your Hugging Face Space repository root:

- `app.py`
- `requirements.txt`

Recommended additional files to copy alongside them:

- `class_names.json`

Optional:

- sample images if you want examples

Notes:

- The app uses the same Google Drive model download strategy as the main project.
- If `class_names.json` is missing, the app falls back to the default class order:
  - `CNV`
  - `DME`
  - `DRUSEN`
  - `NORMAL`

