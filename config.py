import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_env():
    env_path = os.path.join(BASE_DIR, ".env.local")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_local_env()

MODEL_PATH = os.path.join(BASE_DIR, "oct_retinal_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
MODEL_FILE_ID = "1jXrsLBiQHOnukSr1dg3CIeUXFshr05cF"
MODEL_DOWNLOAD_URL = os.environ.get(
    "MODEL_DOWNLOAD_URL",
    f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}",
)
AUTO_DOWNLOAD_MODEL = os.environ.get("AUTO_DOWNLOAD_MODEL", "1").lower() not in {"0", "false", "no"}

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

BASE_URL = "http://127.0.0.1:5000/"

GEMINI_API_BASE_URL = os.environ.get("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
