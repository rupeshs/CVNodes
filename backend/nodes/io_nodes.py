import cv2

from config import OUTPUT_DIR, UPLOAD_DIR
from engine.registry import register_node


@register_node("cv/LoadImage")
class LoadImageNode:
    NAME = "Load Image"
    CATEGORY = "IO"
    INPUTS = []
    OUTPUTS = [{"name": "image", "type": "IMAGE"}]
    # "IMAGE_UPLOAD" is a frontend-only widget kind: it renders a button that
    # opens a file picker, uploads via /api/upload, and stores the returned
    # filename as this widget's value.
    WIDGETS = [{"name": "filename", "type": "IMAGE_UPLOAD", "default": ""}]

    def run(self, filename=""):
        if not filename:
            raise ValueError("No image selected")
        path = UPLOAD_DIR / filename
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read uploaded image '{filename}'")
        return {"image": image}


@register_node("cv/PreviewImage")
class PreviewImageNode:
    NAME = "Preview Image"
    CATEGORY = "IO"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = []
    WIDGETS = []
    IS_OUTPUT_NODE = True

    def run(self, image):
        return {"image": image}


@register_node("cv/SaveImage")
class SaveImageNode:
    NAME = "Save Image"
    CATEGORY = "IO"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = []
    WIDGETS = [{"name": "filename", "type": "STRING", "default": "output.png"}]
    IS_OUTPUT_NODE = True

    def run(self, image, filename="output.png"):
        safe_name = filename.strip() or "output.png"
        path = OUTPUT_DIR / safe_name
        cv2.imwrite(str(path), image)
        return {"image": image}
