import cv2

from engine.registry import register_node


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


@register_node("cv/AdaptiveThreshold")
class AdaptiveThresholdNode:
    NAME = "Adaptive Threshold"
    CATEGORY = "OpenCV/Threshold"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "maxval", "type": "INT", "default": 255, "min": 0, "max": 255, "step": 1},
        {"name": "method", "type": "COMBO", "default": "Gaussian", "options": ["Mean", "Gaussian"]},
        {"name": "mode", "type": "COMBO", "default": "BINARY", "options": ["BINARY", "BINARY_INV"]},
        {"name": "block_size", "type": "INT", "default": 11, "min": 3, "max": 99, "step": 2},
        {"name": "c", "type": "INT", "default": 2, "min": -50, "max": 50, "step": 1},
    ]

    _METHODS = {"Mean": cv2.ADAPTIVE_THRESH_MEAN_C, "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
    _MODES = {"BINARY": cv2.THRESH_BINARY, "BINARY_INV": cv2.THRESH_BINARY_INV}

    def run(self, image, maxval=255, method="Gaussian", mode="BINARY", block_size=11, c=2):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block = int(block_size)
        if block % 2 == 0:
            block += 1
        block = max(3, block)
        adaptive_method = self._METHODS.get(method, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        thresh_type = self._MODES.get(mode, cv2.THRESH_BINARY)
        result = cv2.adaptiveThreshold(gray, maxval, adaptive_method, thresh_type, block, c)
        return {"image": result}
