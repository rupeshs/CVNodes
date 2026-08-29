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
