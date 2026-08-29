"""Helpers to move image data between numpy (OpenCV's native format) and
base64-encoded PNG (what the browser can display directly in an <img> tag)."""

import base64

import cv2
import numpy as np


def numpy_to_base64_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buf).decode("ascii")


def base64_to_numpy(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode base64 image data")
    return image
