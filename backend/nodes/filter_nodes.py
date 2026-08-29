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


@register_node("cv/Sobel")
class SobelNode:
    NAME = "Sobel"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "direction", "type": "COMBO", "default": "Both", "options": ["X", "Y", "Both"]},
        {"name": "ksize", "type": "INT", "default": 3, "min": 1, "max": 7, "step": 2},
    ]

    def run(self, image, direction="Both", ksize=3):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        k = _odd(ksize)
        if direction == "X":
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        elif direction == "Y":
            grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        else:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            grad = cv2.magnitude(gx, gy)
        return {"image": cv2.convertScaleAbs(grad)}


@register_node("cv/Laplacian")
class LaplacianNode:
    NAME = "Laplacian"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "ksize", "type": "INT", "default": 3, "min": 1, "max": 31, "step": 2}]

    def run(self, image, ksize=3):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=_odd(ksize))
        return {"image": cv2.convertScaleAbs(lap)}


@register_node("cv/BilateralFilter")
class BilateralFilterNode:
    NAME = "Bilateral Filter"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "d", "type": "INT", "default": 9, "min": 1, "max": 25, "step": 1},
        {"name": "sigma_color", "type": "FLOAT", "default": 75, "min": 1, "max": 200, "step": 1},
        {"name": "sigma_space", "type": "FLOAT", "default": 75, "min": 1, "max": 200, "step": 1},
    ]

    def run(self, image, d=9, sigma_color=75, sigma_space=75):
        result = cv2.bilateralFilter(image, int(d), float(sigma_color), float(sigma_space))
        return {"image": result}


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


def _kernel(ksize):
    k = max(1, int(ksize))
    return np.ones((k, k), np.uint8)


@register_node("cv/Dilate")
class DilateNode:
    NAME = "Dilate"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "ksize", "type": "INT", "default": 5, "min": 1, "max": 51, "step": 1},
        {"name": "iterations", "type": "INT", "default": 1, "min": 1, "max": 20, "step": 1},
    ]

    def run(self, image, ksize=5, iterations=1):
        result = cv2.dilate(image, _kernel(ksize), iterations=max(1, int(iterations)))
        return {"image": result}


@register_node("cv/Erode")
class ErodeNode:
    NAME = "Erode"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "ksize", "type": "INT", "default": 5, "min": 1, "max": 51, "step": 1},
        {"name": "iterations", "type": "INT", "default": 1, "min": 1, "max": 20, "step": 1},
    ]

    def run(self, image, ksize=5, iterations=1):
        result = cv2.erode(image, _kernel(ksize), iterations=max(1, int(iterations)))
        return {"image": result}


@register_node("cv/Morphology")
class MorphologyNode:
    NAME = "Morphology"
    CATEGORY = "OpenCV/Filters"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {
            "name": "operation",
            "type": "COMBO",
            "default": "Dilate",
            "options": ["Erode", "Dilate", "Open", "Close", "Gradient"],
        },
        {"name": "ksize", "type": "INT", "default": 5, "min": 1, "max": 51, "step": 1},
        {"name": "iterations", "type": "INT", "default": 1, "min": 1, "max": 20, "step": 1},
    ]

    _OPS = {
        "Erode": cv2.MORPH_ERODE,
        "Dilate": cv2.MORPH_DILATE,
        "Open": cv2.MORPH_OPEN,
        "Close": cv2.MORPH_CLOSE,
        "Gradient": cv2.MORPH_GRADIENT,
    }

    def run(self, image, operation="Dilate", ksize=5, iterations=1):
        op = self._OPS.get(operation, cv2.MORPH_DILATE)
        result = cv2.morphologyEx(image, op, _kernel(ksize), iterations=max(1, int(iterations)))
        return {"image": result}
