import cv2

from engine.registry import register_node


@register_node("cv/NoOp")
class NoOpNode:
    NAME = "No Op"
    CATEGORY = "OpenCV/Compositing"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    WIDGETS = []

    def run(self, image):
  
        return {"image": image}
