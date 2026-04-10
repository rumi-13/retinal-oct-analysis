import os
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER

def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filename, filepath