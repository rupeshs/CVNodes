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
