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
