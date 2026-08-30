import cv2
import numpy as np

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
            for x1, y1, x2, y2 in lines[:, 0]:
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
