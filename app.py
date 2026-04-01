"""
app.py — WasteGuard Flask application entry point.

Wires together FileHandler, Detector, and Annotator into HTTP routes.
"""

import logging
import os

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from annotator import annotate
from detector import Detector
from file_handler import allowed_file, ensure_directories, save_upload

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Config:
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "static/outputs")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024)))
    MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = os.getenv("SECRET_KEY", "wasteguard-secret")
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

ensure_directories(Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER)
detector = Detector(Config.MODEL_PATH)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", report=None)

    # --- POST ---
    file = request.files.get("file")

    if not file or file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
        flash("Invalid file type. Please upload a JPG, JPEG, or PNG image.")
        return redirect(url_for("index"))

    logger.info("Detection request received: %s", file.filename)

    input_path = save_upload(file, Config.UPLOAD_FOLDER)
    unique_input_filename = os.path.basename(input_path)

    # Output shares the same UUID-prefixed filename as the input
    output_filename = unique_input_filename
    output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)

    try:
        result = detector.detect(input_path)
        annotate(input_path, result, output_path)
    except Exception as e:
        logger.error("Inference failed for %s: %s", input_path, e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    incident_status = (
        "Incident Detected"
        if (result.person.found or result.waste.found or result.plate.found)
        else "No Incident"
    )

    report = {
        "total_objects": result.total_objects,
        "incident_status": incident_status,
        "person": {"found": result.person.found, "count": result.person.count},
        "waste": {"found": result.waste.found, "count": result.waste.count},
        "plate": {"found": result.plate.found, "count": result.plate.count},
        "input_url": url_for("uploaded_file", filename=unique_input_filename),
        "output_url": url_for("output_file", filename=output_filename),
    }

    logger.info(
        "Detection complete for %s — status: %s, total objects: %d",
        unique_input_filename,
        incident_status,
        result.total_objects,
    )

    return render_template("index.html", report=report)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(Config.OUTPUT_FOLDER, filename)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(413)
def request_entity_too_large(e):
    flash("File too large. Maximum size is 10 MB.")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=Config.FLASK_DEBUG)
