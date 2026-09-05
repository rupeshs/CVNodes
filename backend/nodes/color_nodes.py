import cv2
import numpy as np

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


_COLOR_CODES = {
    "BGR to HSV": cv2.COLOR_BGR2HSV,
    "HSV to BGR": cv2.COLOR_HSV2BGR,
    "BGR to LAB": cv2.COLOR_BGR2LAB,
    "LAB to BGR": cv2.COLOR_LAB2BGR,
    "BGR to YCrCb": cv2.COLOR_BGR2YCrCb,
    "YCrCb to BGR": cv2.COLOR_YCrCb2BGR,
    "BGR to RGB": cv2.COLOR_BGR2RGB,
    "BGR to Gray": cv2.COLOR_BGR2GRAY,
    "Gray to BGR": cv2.COLOR_GRAY2BGR,
}


@register_node("cv/ColorConvert")
class ColorConvertNode:
    NAME = "Color Convert"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "mode", "type": "COMBO", "default": "BGR to HSV", "options": list(_COLOR_CODES)}
    ]

    def run(self, image, mode="BGR to HSV"):
        code = _COLOR_CODES.get(mode)
        if code is None:
            return {"image": image}

        src = image
        if mode == "Gray to BGR" and src.ndim == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        elif mode == "BGR to Gray" and src.ndim == 2:
            return {"image": src}
        elif mode not in ("Gray to BGR", "BGR to Gray") and src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        return {"image": cv2.cvtColor(src, code)}


@register_node("cv/InRange")
class InRangeNode:
    NAME = "Color Range Mask"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "mask", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "low1", "type": "INT", "default": 0, "min": 0, "max": 255, "step": 1},
        {"name": "low2", "type": "INT", "default": 0, "min": 0, "max": 255, "step": 1},
        {"name": "low3", "type": "INT", "default": 0, "min": 0, "max": 255, "step": 1},
        {"name": "high1", "type": "INT", "default": 255, "min": 0, "max": 255, "step": 1},
        {"name": "high2", "type": "INT", "default": 255, "min": 0, "max": 255, "step": 1},
        {"name": "high3", "type": "INT", "default": 255, "min": 0, "max": 255, "step": 1},
    ]

    def run(self, image, low1=0, low2=0, low3=0, high1=255, high2=255, high3=255):
        if image.ndim == 2:
            return {"mask": cv2.inRange(image, low1, high1)}
        lower = np.array([low1, low2, low3], dtype=np.uint8)
        upper = np.array([high1, high2, high3], dtype=np.uint8)
        return {"mask": cv2.inRange(image, lower, upper)}


@register_node("cv/SplitChannels")
class SplitChannelsNode:
    NAME = "Split Channels"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [
        {"name": "channel_0", "type": "IMAGE"},
        {"name": "channel_1", "type": "IMAGE"},
        {"name": "channel_2", "type": "IMAGE"},
    ]
    WIDGETS = []

    def run(self, image):
        if image.ndim == 2:
            return {"channel_0": image, "channel_1": image, "channel_2": image}
        b, g, r = cv2.split(image)
        return {"channel_0": b, "channel_1": g, "channel_2": r}


@register_node("cv/MergeChannels")
class MergeChannelsNode:
    NAME = "Merge Channels"
    CATEGORY = "OpenCV/Color"
    INPUTS = [
        {"name": "channel_0", "type": "IMAGE"},
        {"name": "channel_1", "type": "IMAGE"},
        {"name": "channel_2", "type": "IMAGE"},
    ]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = []

    def run(self, channel_0, channel_1, channel_2):
        return {"image": cv2.merge([channel_0, channel_1, channel_2])}


@register_node("cv/HistogramEqualize")
class HistogramEqualizeNode:
    NAME = "Histogram Equalize"
    CATEGORY = "OpenCV/Color"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [
        {"name": "method", "type": "COMBO", "default": "CLAHE", "options": ["Equalize", "CLAHE"]},
        {"name": "clip_limit", "type": "FLOAT", "default": 2.0, "min": 0.1, "max": 40.0, "step": 0.1},
        {"name": "tile_size", "type": "INT", "default": 8, "min": 1, "max": 64, "step": 1},
    ]

    def _equalize(self, channel, method, clip_limit, tile_size):
        if method == "Equalize":
            return cv2.equalizeHist(channel)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(tile_size), int(tile_size)))
        return clahe.apply(channel)

    def run(self, image, method="CLAHE", clip_limit=2.0, tile_size=8):
        if image.ndim == 2:
            return {"image": self._equalize(image, method, clip_limit, tile_size)}

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = self._equalize(y, method, clip_limit, tile_size)
        result = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
        return {"image": result}
