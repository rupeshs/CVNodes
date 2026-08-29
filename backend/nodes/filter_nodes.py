import cv2

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
