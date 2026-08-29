import cv2
import numpy as np

from engine.registry import register_node


def _odd(n):
    """Kernel sizes in OpenCV must be positive and odd."""
    n = int(n)
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return n


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


@register_node("cv/GaussianBlur")
class GaussianBlurNode:
    NAME = "Gaussian Blur"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "ksize", "type": "INT", "default": 5, "min": 1, "max": 51, "step": 2},
        {"name": "sigma", "type": "FLOAT", "default": 0, "min": 0, "max": 50, "step": 0.5},
    ]

    def run(self, image, ksize=5, sigma=0):
        k = _odd(ksize)
        return {"image": cv2.GaussianBlur(image, (k, k), sigma)}


@register_node("cv/MedianBlur")
class MedianBlurNode:
    NAME = "Median Blur"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "ksize", "type": "INT", "default": 5, "min": 1, "max": 51, "step": 2}]

    def run(self, image, ksize=5):
        return {"image": cv2.medianBlur(image, _odd(ksize))}


@register_node("cv/CannyEdge")
class CannyEdgeNode:
    NAME = "Canny Edge Detection"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "threshold1", "type": "FLOAT", "default": 100, "min": 0, "max": 500, "step": 1},
        {"name": "threshold2", "type": "FLOAT", "default": 200, "min": 0, "max": 500, "step": 1},
    ]

    def run(self, image, threshold1=100, threshold2=200):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {"image": cv2.Canny(gray, threshold1, threshold2)}


@register_node("cv/Threshold")
class ThresholdNode:
    NAME = "Threshold"
    CATEGORY = "OpenCV/Threshold"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "thresh", "type": "INT", "default": 127, "min": 0, "max": 255, "step": 1},
        {"name": "maxval", "type": "INT", "default": 255, "min": 0, "max": 255, "step": 1},
        {
            "name": "mode",
            "type": "COMBO",
            "default": "BINARY",
            "options": ["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV", "OTSU"],
        },
    ]

    _MODES = {
        "BINARY": cv2.THRESH_BINARY,
        "BINARY_INV": cv2.THRESH_BINARY_INV,
        "TRUNC": cv2.THRESH_TRUNC,
        "TOZERO": cv2.THRESH_TOZERO,
        "TOZERO_INV": cv2.THRESH_TOZERO_INV,
        "OTSU": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    }

    def run(self, image, thresh=127, maxval=255, mode="BINARY"):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flag = self._MODES.get(mode, cv2.THRESH_BINARY)
        _ret, result = cv2.threshold(gray, thresh, maxval, flag)
        return {"image": result}


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


@register_node("cv/Sharpen")
class SharpenNode:
    NAME = "Sharpen"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "amount", "type": "FLOAT", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}]

    def run(self, image, amount=1.0):
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return {"image": sharpened}


@register_node("cv/Blend")
class BlendNode:
    NAME = "Blend"
    CATEGORY = "OpenCV/Compositing"
    INPUTS = [{"name": "image_a", "type": "IMAGE"}, {"name": "image_b", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "alpha", "type": "FLOAT", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}]

    def run(self, image_a, image_b, alpha=0.5):
        if image_a.shape[:2] != image_b.shape[:2]:
            image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]))
        if image_a.ndim != image_b.ndim:
            if image_a.ndim == 2:
                image_a = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
            if image_b.ndim == 2:
                image_b = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(image_a, alpha, image_b, 1 - alpha, 0)
        return {"image": blended}
