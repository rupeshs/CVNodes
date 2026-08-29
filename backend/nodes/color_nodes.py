import cv2

from engine.registry import register_node


@register_node("cv/Grayscale")
class GrayscaleNode:
    NAME = "Grayscale"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = []

    def run(self, image):
        if image.ndim == 2:
            return {"image": image}
        return {"image": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)}


@register_node("cv/InvertColors")
class InvertColorsNode:
    NAME = "Invert Colors"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = []

    def run(self, image):
        return {"image": cv2.bitwise_not(image)}


@register_node("cv/BrightnessContrast")
class BrightnessContrastNode:
    NAME = "Brightness / Contrast"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "brightness", "type": "INT", "default": 0, "min": -255, "max": 255, "step": 1},
        {"name": "contrast", "type": "FLOAT", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
    ]

    def run(self, image, brightness=0, contrast=1.0):
        result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return {"image": result}
