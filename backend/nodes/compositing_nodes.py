import cv2

from engine.registry import register_node


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


_BITWISE_OPS = {"AND": cv2.bitwise_and, "OR": cv2.bitwise_or, "XOR": cv2.bitwise_xor}


@register_node("cv/BitwiseOp")
class BitwiseOpNode:
    NAME = "Bitwise Op"
    CATEGORY = "OpenCV/Compositing"
    INPUTS = [{"name": "image_a", "type": "IMAGE"}, {"name": "image_b", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = [{"name": "operation", "type": "COMBO", "default": "AND", "options": list(_BITWISE_OPS)}]

    def run(self, image_a, image_b, operation="AND"):
        if image_a.shape[:2] != image_b.shape[:2]:
            image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]))
        if image_a.ndim != image_b.ndim:
            if image_a.ndim == 2:
                image_a = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
            if image_b.ndim == 2:
                image_b = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)
        op = _BITWISE_OPS.get(operation, cv2.bitwise_and)
        return {"image": op(image_a, image_b)}


@register_node("cv/ApplyMask")
class ApplyMaskNode:
    NAME = "Apply Mask"
    CATEGORY = "OpenCV/Compositing"
    INPUTS = [{"name": "image", "type": "IMAGE"}, {"name": "mask", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = []

    def run(self, image, mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        return {"image": cv2.bitwise_and(image, image, mask=mask)}
