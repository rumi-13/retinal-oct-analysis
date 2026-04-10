import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "oct_retinal_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
MODEL_FILE_ID = "1jXrsLBiQHOnukSr1dg3CIeUXFshr05cF"
MODEL_DOWNLOAD_URL = os.environ.get(
    "MODEL_DOWNLOAD_URL",
    f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}",
)
AUTO_DOWNLOAD_MODEL = os.environ.get("AUTO_DOWNLOAD_MODEL", "1").lower() not in {"0", "false", "no"}

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")

BASE_URL = "http://127.0.0.1:5000/"
