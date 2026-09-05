import urllib.request

import cv2
import numpy as np

from config import MODELS_DIR
from engine.registry import register_node


@register_node("cv/HoughLines")
class HoughLinesNode:
    NAME = "Hough Lines"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "threshold", "type": "INT", "default": 80, "min": 1, "max": 300, "step": 1},
        {"name": "min_line_length", "type": "INT", "default": 50, "min": 0, "max": 500, "step": 1},
        {"name": "max_line_gap", "type": "INT", "default": 10, "min": 0, "max": 100, "step": 1},
    ]

    def run(self, image, threshold=80, min_line_length=50, max_line_gap=10):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canvas = image.copy()
        else:
            gray = image
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(
            gray,
            1,
            np.pi / 180,
            threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is not None:
            # cv2.HoughLinesP's return shape changed across opencv-python
            # versions ((N, 1, 4) vs (N, 4)) - reshape(-1, 4) works for both.
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return {"image": canvas}


_CONTOUR_MODES = {
    "EXTERNAL": cv2.RETR_EXTERNAL,
    "LIST": cv2.RETR_LIST,
    "TREE": cv2.RETR_TREE,
    "CCOMP": cv2.RETR_CCOMP,
}

_CONTOUR_METHODS = {
    "SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
    "NONE": cv2.CHAIN_APPROX_NONE,
}


@register_node("cv/FindContours")
class FindContoursNode:
    NAME = "Find Contours"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [
        {"name": "image", "type": "IMAGE"},
        {"name": "contours", "type": "CONTOURS"},
    ]
    WIDGETS = [
        {"name": "mode", "type": "COMBO", "default": "EXTERNAL", "options": list(_CONTOUR_MODES)},
        {"name": "method", "type": "COMBO", "default": "SIMPLE", "options": list(_CONTOUR_METHODS)},
        {"name": "min_area", "type": "INT", "default": 0, "min": 0, "max": 100000, "step": 10},
        {"name": "thickness", "type": "INT", "default": 2, "min": 1, "max": 20, "step": 1},
    ]

    def run(self, image, mode="EXTERNAL", method="SIMPLE", min_area=0, thickness=2):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canvas = image.copy()
        else:
            gray = image
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        found, _hierarchy = cv2.findContours(
            gray, _CONTOUR_MODES[mode], _CONTOUR_METHODS[method]
        )

        contours = []
        for c in found:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            contours.append(
                {
                    "area": float(area),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "points": c.reshape(-1, 2).tolist(),
                }
            )

        cv2.drawContours(
            canvas,
            [np.array(c["points"], dtype=np.int32) for c in contours],
            -1,
            (0, 255, 0),
            thickness,
        )

        return {"image": canvas, "contours": contours}


def _order_quad_points(pts):
    """Orders 4 points as [top-left, top-right, bottom-right, bottom-left],
    the corner order cv2.getPerspectiveTransform's dst rectangle expects."""
    pts = np.array(pts, dtype=np.float32)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left = pts[np.argmin(total)]
    bottom_right = pts[np.argmax(total)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return [top_left.tolist(), top_right.tolist(), bottom_right.tolist(), bottom_left.tolist()]


@register_node("cv/ApproxQuad")
class ApproxQuadNode:
    NAME = "Approx Quad"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "contours", "type": "CONTOURS"}]
    OUTPUTS = [{"name": "corners", "type": "CORNERS"}]
    WIDGETS = [
        {"name": "epsilon", "type": "FLOAT", "default": 0.02, "min": 0.001, "max": 0.2, "step": 0.001},
    ]

    def run(self, contours, epsilon=0.02):
        if not contours:
            raise ValueError("No contours to approximate")

        largest = max(contours, key=lambda c: c["area"])
        pts = np.array(largest["points"], dtype=np.int32).reshape(-1, 1, 2)
        peri = cv2.arcLength(pts, True)
        approx = cv2.approxPolyDP(pts, epsilon * peri, True)

        if len(approx) != 4:
            raise ValueError(
                f"Largest contour approximated to {len(approx)} points, not 4 - "
                "adjust epsilon or feed a more rectangular contour"
            )

        return {"corners": _order_quad_points(approx.reshape(-1, 2))}


_BOX_COLORS = {
    "Blue": (255, 0, 0),
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Yellow": (0, 255, 255),
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255),
}


@register_node("cv/BoundingBoxes")
class BoundingBoxesNode:
    NAME = "Bounding Boxes"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [
        {"name": "image", "type": "IMAGE"},
        {"name": "contours", "type": "CONTOURS"},
    ]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "color", "type": "COMBO", "default": "Blue", "options": list(_BOX_COLORS)},
        {"name": "thickness", "type": "INT", "default": 2, "min": 1, "max": 20, "step": 1},
    ]

    def run(self, image, contours, color="Blue", thickness=2):
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bgr = _BOX_COLORS.get(color, (255, 0, 0))

        for c in contours or []:
            x, y, w, h = c["bbox"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr, thickness)

        return {"image": canvas}


@register_node("cv/HoughCircles")
class HoughCirclesNode:
    NAME = "Hough Circles"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "dp", "type": "FLOAT", "default": 1.2, "min": 0.1, "max": 5.0, "step": 0.1},
        {"name": "min_dist", "type": "INT", "default": 20, "min": 1, "max": 500, "step": 1},
        {"name": "param1", "type": "FLOAT", "default": 100, "min": 1, "max": 500, "step": 1},
        {"name": "param2", "type": "FLOAT", "default": 30, "min": 1, "max": 500, "step": 1},
        {"name": "min_radius", "type": "INT", "default": 0, "min": 0, "max": 500, "step": 1},
        {"name": "max_radius", "type": "INT", "default": 0, "min": 0, "max": 500, "step": 1},
    ]

    def run(
        self,
        image,
        dp=1.2,
        min_dist=20,
        param1=100,
        param2=30,
        min_radius=0,
        max_radius=0,
    ):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canvas = image.copy()
        else:
            gray = image
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)
                cv2.circle(canvas, (x, y), 2, (0, 0, 255), 3)

        return {"image": canvas}


_TM_METHODS = {
    "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
}


@register_node("cv/TemplateMatch")
class TemplateMatchNode:
    NAME = "Template Match"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "image", "type": "IMAGE"}, {"name": "template", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {
            "name": "method",
            "type": "COMBO",
            "default": "CCOEFF_NORMED",
            "options": list(_TM_METHODS),
        },
        {"name": "threshold", "type": "FLOAT", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    ]

    def run(self, image, template, method="CCOEFF_NORMED", threshold=0.8):
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmpl = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        th, tw = tmpl.shape[:2]

        flag = _TM_METHODS.get(method, cv2.TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(gray, tmpl, flag)

        if flag == cv2.TM_SQDIFF_NORMED:
            locations = np.where(result <= (1 - threshold))
        else:
            locations = np.where(result >= threshold)

        for y, x in zip(*locations):
            cv2.rectangle(canvas, (int(x), int(y)), (int(x) + tw, int(y) + th), (0, 255, 0), 2)

        return {"image": canvas}


@register_node("cv/ConnectedComponents")
class ConnectedComponentsNode:
    NAME = "Connected Components"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "connectivity", "type": "COMBO", "default": "8", "options": ["4", "8"]},
        {"name": "min_area", "type": "INT", "default": 0, "min": 0, "max": 100000, "step": 10},
    ]

    def run(self, image, connectivity="8", min_area=0):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=int(connectivity)
        )

        colors = np.random.RandomState(42).randint(0, 255, size=(num_labels, 3)).astype(np.uint8)
        colors[0] = 0
        canvas = colors[labels]

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                canvas[labels == i] = 0

        return {"image": canvas}


# Newer opencv-python builds (5.x) dropped the CascadeClassifier /
# haarcascades data files from the Python package, so face detection here
# uses the DNN-based YuNet detector that ships as part of core cv2 instead.
# Its weights aren't bundled either - they're downloaded once to MODELS_DIR
# and cached on disk (same "first run downloads a model" pattern already
# used by the RemoveBackground custom node).
_YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
_YUNET_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_FACE_DETECTOR = None


def _get_face_detector(width, height, score_threshold):
    global _FACE_DETECTOR
    model_path = MODELS_DIR / _YUNET_MODEL_NAME
    if not model_path.exists():
        tmp_path = model_path.with_suffix(".onnx.part")
        urllib.request.urlretrieve(_YUNET_MODEL_URL, tmp_path)
        tmp_path.rename(model_path)

    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = cv2.FaceDetectorYN_create(str(model_path), "", (width, height))
    _FACE_DETECTOR.setInputSize((width, height))
    _FACE_DETECTOR.setScoreThreshold(score_threshold)
    return _FACE_DETECTOR


@register_node("cv/FaceDetect")
class FaceDetectNode:
    NAME = "Face Detect"
    CATEGORY = "OpenCV/Detection"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "score_threshold", "type": "FLOAT", "default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05},
    ]

    def run(self, image, score_threshold=0.6):
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = canvas.shape[:2]

        detector = _get_face_detector(w, h, score_threshold)
        _ret, faces = detector.detect(canvas)
        for face in faces if faces is not None else []:
            x, y, fw, fh = face[:4].astype(int)
            cv2.rectangle(canvas, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        return {"image": canvas}
