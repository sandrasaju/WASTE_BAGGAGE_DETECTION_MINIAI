"""Unit tests for file_handler.py."""

import pytest
from file_handler import allowed_file, generate_unique_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# --- allowed_file: valid extensions ---

def test_allowed_file_jpg():
    assert allowed_file("photo.jpg", ALLOWED_EXTENSIONS) is True


def test_allowed_file_jpeg_uppercase():
    assert allowed_file("image.JPEG", ALLOWED_EXTENSIONS) is True


def test_allowed_file_png_uppercase():
    assert allowed_file("file.PNG", ALLOWED_EXTENSIONS) is True


# --- allowed_file: invalid extensions ---

def test_allowed_file_exe():
    assert allowed_file("virus.exe", ALLOWED_EXTENSIONS) is False


def test_allowed_file_gif():
    assert allowed_file("anim.gif", ALLOWED_EXTENSIONS) is False


def test_allowed_file_no_extension():
    assert allowed_file("noext", ALLOWED_EXTENSIONS) is False


# --- generate_unique_filename ---

def test_generate_unique_filename_contains_sanitised_name():
    result = generate_unique_filename("photo.jpg")
    assert "photo.jpg" in result


def test_generate_unique_filename_two_calls_differ():
    name1 = generate_unique_filename("photo.jpg")
    name2 = generate_unique_filename("photo.jpg")
    assert name1 != name2


# ---------------------------------------------------------------------------
# Tests for detector.py
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch
from detector import map_label_to_category, TargetResult, DetectionResult


# --- map_label_to_category ---

def test_map_label_person():
    assert map_label_to_category("person") == "person"


def test_map_label_bottle_is_waste():
    assert map_label_to_category("bottle") == "waste"


def test_map_label_car_is_plate():
    assert map_label_to_category("car") == "plate"


def test_map_label_unknown():
    assert map_label_to_category("unknown_thing") == "unknown"


def test_map_label_case_insensitive():
    assert map_label_to_category("PERSON") == "person"


# --- TargetResult found/count invariant ---

def test_target_result_count_zero_found_false():
    t = TargetResult(found=False, count=0)
    assert t.found is False
    assert t.count == 0


def test_target_result_count_nonzero_found_true():
    t = TargetResult(found=True, count=3)
    assert t.found is True
    assert t.count == 3


# --- DetectionResult total_objects ---

def test_detection_result_total_objects():
    person = TargetResult(found=True, count=1)
    waste = TargetResult(found=True, count=2)
    plate = TargetResult(found=True, count=1)
    result = DetectionResult(person=person, waste=waste, plate=plate, total_objects=4)
    assert result.total_objects == 4
    assert result.total_objects == person.count + waste.count + plate.count


# --- Detector.detect with mocked YOLO ---

def _make_mock_box(xyxy, conf, cls_idx):
    box = MagicMock()
    xyxy_tensor = MagicMock()
    xyxy_tensor.tolist.return_value = xyxy
    box.xyxy = [xyxy_tensor]
    box.conf = [conf]
    box.cls = [cls_idx]
    return box


def test_detector_detect_one_person():
    """Mock YOLO returning one 'person' box → person.found True, count 1."""
    mock_box = _make_mock_box([0.0, 0.0, 10.0, 10.0], 0.9, 0)
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}

    with patch("detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        # Patch os.path.isfile so __init__ doesn't sys.exit
        with patch("detector.os.path.isfile", return_value=True):
            from detector import Detector
            det = Detector("fake_model.pt")
            detection = det.detect("fake_image.jpg")

    assert detection.person.found is True
    assert detection.person.count == 1


def test_detector_detect_empty_boxes():
    """Mock YOLO returning no boxes → all found=False, total_objects=0."""
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_result.names = {}

    with patch("detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        with patch("detector.os.path.isfile", return_value=True):
            from detector import Detector
            det = Detector("fake_model.pt")
            detection = det.detect("fake_image.jpg")

    assert detection.person.found is False
    assert detection.waste.found is False
    assert detection.plate.found is False
    assert detection.total_objects == 0


# ---------------------------------------------------------------------------
# Tests for annotator.py
# ---------------------------------------------------------------------------

import filecmp
import numpy as np
import cv2

from annotator import annotate
from detector import DetectionResult, TargetResult


def _empty_target() -> TargetResult:
    return TargetResult(found=False, count=0)


def _make_synthetic_image(path):
    """Write a 100x100 black PNG to *path* using numpy/cv2."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# --- annotate with one person box ---

def test_annotate_with_person_box_output_exists(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _make_synthetic_image(input_path)

    person = TargetResult(
        found=True,
        count=1,
        boxes=[[10, 10, 50, 50]],
        scores=[0.9],
        labels=["person"],
    )
    result = DetectionResult(
        person=person,
        waste=_empty_target(),
        plate=_empty_target(),
        total_objects=1,
    )

    annotate(str(input_path), result, str(output_path))

    assert output_path.exists()
    out_img = cv2.imread(str(output_path))
    assert out_img.shape[:2] == (100, 100)


# --- annotate with zero detections ---

def test_annotate_zero_detections_output_exists(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _make_synthetic_image(input_path)

    result = DetectionResult(
        person=_empty_target(),
        waste=_empty_target(),
        plate=_empty_target(),
        total_objects=0,
    )

    annotate(str(input_path), result, str(output_path))

    assert output_path.exists()


# --- zero-detection path: output identical to input ---

def test_annotate_zero_detections_output_identical_to_input(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _make_synthetic_image(input_path)

    result = DetectionResult(
        person=_empty_target(),
        waste=_empty_target(),
        plate=_empty_target(),
        total_objects=0,
    )

    annotate(str(input_path), result, str(output_path))

    assert filecmp.cmp(str(input_path), str(output_path), shallow=False)


# ---------------------------------------------------------------------------
# Flask route tests
# ---------------------------------------------------------------------------

import io
import os
import struct
import zlib
from unittest.mock import patch, MagicMock

import pytest

import app as app_module
from app import app as flask_app
from detector import DetectionResult, TargetResult


def _empty_target_r() -> TargetResult:
    return TargetResult(found=False, count=0)


def _make_dummy_detection() -> DetectionResult:
    return DetectionResult(
        person=_empty_target_r(),
        waste=_empty_target_r(),
        plate=_empty_target_r(),
        total_objects=0,
    )


def _minimal_png() -> bytes:
    """Return a valid 1x1 white PNG as bytes without PIL."""
    def _chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00\xff\xff\xff"  # filter byte + RGB white
    idat = _chunk(b"IDAT", zlib.compress(raw_row))
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


@pytest.fixture
def client(tmp_path):
    flask_app.config["TESTING"] = True
    flask_app.config["UPLOAD_FOLDER"] = str(tmp_path / "uploads")
    flask_app.config["OUTPUT_FOLDER"] = str(tmp_path / "outputs")
    os.makedirs(flask_app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(flask_app.config["OUTPUT_FOLDER"], exist_ok=True)
    with flask_app.test_client() as c:
        yield c


def test_route_get_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_route_post_no_file_redirects(client):
    response = client.post("/", data={})
    assert response.status_code == 302


def test_route_post_invalid_extension_redirects(client):
    data = {"file": (io.BytesIO(b"GIF89a"), "animation.gif")}
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 302


def test_route_post_valid_image_returns_200(client):
    dummy_result = _make_dummy_detection()
    dummy_output = str(flask_app.config["OUTPUT_FOLDER"]) + "/out.png"

    with patch.object(app_module.detector, "detect", return_value=dummy_result), \
         patch("app.annotate", return_value=dummy_output), \
         patch("app.save_upload") as mock_save:
        # save_upload must return a path inside the upload folder
        mock_save.return_value = flask_app.config["UPLOAD_FOLDER"] + "/test_img.png"

        data = {"file": (io.BytesIO(_minimal_png()), "test.png")}
        response = client.post("/", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    body = response.data.decode("utf-8", errors="replace")
    assert "WasteGuard" in body or "report" in body.lower()


def test_route_inference_exception_returns_500(client):
    with patch.object(app_module.detector, "detect", side_effect=Exception("model crash")), \
         patch("app.save_upload") as mock_save:
        mock_save.return_value = flask_app.config["UPLOAD_FOLDER"] + "/test_img.png"

        data = {"file": (io.BytesIO(_minimal_png()), "test.png")}
        response = client.post("/", data=data, content_type="multipart/form-data")

    assert response.status_code == 500
    json_body = response.get_json()
    assert json_body is not None
    assert "error" in json_body
