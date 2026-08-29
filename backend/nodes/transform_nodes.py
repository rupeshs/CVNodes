import cv2

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
