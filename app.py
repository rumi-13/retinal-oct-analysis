from flask import Flask
from flask_cors import CORS
import os

from routes.predict import predict_bp
from config import UPLOAD_FOLDER

app = Flask(__name__, static_folder='static')
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.register_blueprint(predict_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)