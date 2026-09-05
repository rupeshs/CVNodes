import cv2
import numpy as np

from engine.registry import register_node


@register_node("cv/Resize")
class ResizeNode:
    NAME = "Resize"
    CATEGORY = "OpenCV/Transform"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "width", "type": "INT", "default": 512, "min": 1, "max": 8192, "step": 1},
        {"name": "height", "type": "INT", "default": 512, "min": 1, "max": 8192, "step": 1},
    ]

    def run(self, image, width=512, height=512):
        return {"image": cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)}


@register_node("cv/Rotate")
class RotateNode:
    NAME = "Rotate"
    CATEGORY = "OpenCV/Transform"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "angle", "type": "FLOAT", "default": 90, "min": -360, "max": 360, "step": 1}]

    def run(self, image, angle=90):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return {"image": cv2.warpAffine(image, matrix, (w, h))}


@register_node("cv/Flip")
class FlipNode:
    NAME = "Flip"
    CATEGORY = "OpenCV/Transform"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {
            "name": "direction",
            "type": "COMBO",
            "default": "Horizontal",
            "options": ["Horizontal", "Vertical", "Both"],
        }
    ]

    _CODES = {"Horizontal": 1, "Vertical": 0, "Both": -1}

    def run(self, image, direction="Horizontal"):
        return {"image": cv2.flip(image, self._CODES.get(direction, 1))}


@register_node("cv/Crop")
class CropNode:
    NAME = "Crop"
    CATEGORY = "OpenCV/Transform"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "x", "type": "INT", "default": 0, "min": 0, "max": 8192, "step": 1},
        {"name": "y", "type": "INT", "default": 0, "min": 0, "max": 8192, "step": 1},
        {"name": "width", "type": "INT", "default": 100, "min": 1, "max": 8192, "step": 1},
        {"name": "height", "type": "INT", "default": 100, "min": 1, "max": 8192, "step": 1},
    ]

    def run(self, image, x=0, y=0, width=100, height=100):
        h, w = image.shape[:2]
        x1 = max(0, min(int(x), w - 1))
        y1 = max(0, min(int(y), h - 1))
        x2 = max(x1 + 1, min(x1 + int(width), w))
        y2 = max(y1 + 1, min(y1 + int(height), h))
        return {"image": image[y1:y2, x1:x2]}


@register_node("cv/PerspectiveWarp")
class PerspectiveWarpNode:
    NAME = "Perspective Warp"
    CATEGORY = "OpenCV/Transform"
    INPUTS = [
        {"name": "image", "type": "IMAGE"},
        {"name": "corners", "type": "CORNERS"},
    ]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    # The widget corners are given as % of image width/height, so the
    # defaults (the image's own corners) work for any input size without
    # knowing it ahead of time - drag a corner inward/outward to correct
    # perspective distortion. If the "corners" input is connected (e.g. from
    # an Approx Quad node), those absolute pixel points are used instead and
    # the widgets are ignored.
    WIDGETS = [
        {"name": "tl_x", "type": "FLOAT", "default": 0, "min": -50, "max": 150, "step": 1},
        {"name": "tl_y", "type": "FLOAT", "default": 0, "min": -50, "max": 150, "step": 1},
        {"name": "tr_x", "type": "FLOAT", "default": 100, "min": -50, "max": 150, "step": 1},
        {"name": "tr_y", "type": "FLOAT", "default": 0, "min": -50, "max": 150, "step": 1},
        {"name": "br_x", "type": "FLOAT", "default": 100, "min": -50, "max": 150, "step": 1},
        {"name": "br_y", "type": "FLOAT", "default": 100, "min": -50, "max": 150, "step": 1},
        {"name": "bl_x", "type": "FLOAT", "default": 0, "min": -50, "max": 150, "step": 1},
        {"name": "bl_y", "type": "FLOAT", "default": 100, "min": -50, "max": 150, "step": 1},
    ]

    def run(
        self,
        image,
        corners=None,
        tl_x=0,
        tl_y=0,
        tr_x=100,
        tr_y=0,
        br_x=100,
        br_y=100,
        bl_x=0,
        bl_y=100,
    ):
        h, w = image.shape[:2]
        if corners:
            src = np.float32(corners)
        else:
            src = np.float32(
                [
                    [tl_x / 100 * w, tl_y / 100 * h],
                    [tr_x / 100 * w, tr_y / 100 * h],
                    [br_x / 100 * w, br_y / 100 * h],
                    [bl_x / 100 * w, bl_y / 100 * h],
                ]
            )
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return {"image": cv2.warpPerspective(image, matrix, (w, h))}
