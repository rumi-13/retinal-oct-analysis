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
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
