import cv2

from engine.registry import register_node

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


@register_node("cv/RemoveBackground")
class RemoveBackgroundNode:
    NAME = "Remove Background"
    CATEGORY = "OpenCV/Compositing"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "image", "type": "IMAGE"}, {"name": "mask", "type": "IMAGE"}]
    WIDGETS = []

    def run(self, image):
        from rembg import remove

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cutout_rgba = remove(rgb, session=_get_session())
        alpha = cutout_rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        return {
            "image": cv2.cvtColor(cutout_rgba, cv2.COLOR_RGBA2BGRA),
            "mask": mask,
        }
