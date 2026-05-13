from flask import Flask, send_from_directory
from flask_cors import CORS
import os

from routes.predict import predict_bp
from config import OUTPUT_FOLDER, UPLOAD_FOLDER

app = Flask(__name__, static_folder='static')
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.register_blueprint(predict_bp)


@app.route("/output/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
