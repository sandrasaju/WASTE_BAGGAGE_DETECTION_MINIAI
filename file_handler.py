"""
file_handler.py — Filesystem utilities for WasteGuard.

Handles directory creation, unique filename generation, file validation,
and saving uploaded files to disk.
"""

import os
import uuid

from werkzeug.utils import secure_filename


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Return True if filename has an extension in allowed_extensions (case-insensitive)."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in {e.lower() for e in allowed_extensions}


def ensure_directories(upload_folder: str, output_folder: str) -> None:
    """Create upload and output directories if they do not already exist."""
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)


def generate_unique_filename(original_filename: str) -> str:
    """Return a unique filename: {uuid4_hex}_{secure_filename}."""
    safe_name = secure_filename(original_filename)
    return f"{uuid.uuid4().hex}_{safe_name}"


def save_upload(file_storage, upload_folder: str) -> str:
    """Save a Werkzeug FileStorage object to upload_folder and return the full saved path."""
    unique_name = generate_unique_filename(file_storage.filename)
    save_path = os.path.join(upload_folder, unique_name)
    file_storage.save(save_path)
    return save_path
