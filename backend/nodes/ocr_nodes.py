import urllib.request

import cv2

from config import MODELS_DIR
from engine.registry import register_node

# CRNN text recognizer from the OpenCV Zoo:
# https://github.com/opencv/opencv_zoo/tree/main/models/text_recognition_crnn
# Same "first run downloads a model, cached after that" pattern as FaceDetect's
# YuNet weights. The vocabulary below is the model's *trained* output classes,
# in order - it must match exactly (you can't shrink it to "just digits" to
# bias decoding; that silently relabels the wrong softmax indices).
_CRNN_MODEL_NAME = "text_recognition_CRNN_EN_2021sep.onnx"
_CRNN_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx"
)
_CRNN_VOCABULARY = "0123456789abcdefghijklmnopqrstuvwxyz"
_CRNN_MODEL = None


def _get_crnn_model():
    global _CRNN_MODEL
    if _CRNN_MODEL is not None:
        return _CRNN_MODEL

    model_path = MODELS_DIR / _CRNN_MODEL_NAME
    if not model_path.exists():
        tmp_path = model_path.with_suffix(".onnx.part")
        urllib.request.urlretrieve(_CRNN_MODEL_URL, tmp_path)
        tmp_path.rename(model_path)

    net = cv2.dnn.readNet(str(model_path))
    model = cv2.dnn.TextRecognitionModel(net)
    model.setDecodeType("CTC-greedy")
    model.setVocabulary(list(_CRNN_VOCABULARY))
    # Model expects a 100x32 grayscale crop; TextRecognitionModel resizes/
    # normalizes internally from these params, so callers can pass any size.
    model.setInputParams(1.0 / 127.5, (100, 32), (127.5,))
    _CRNN_MODEL = model
    return _CRNN_MODEL


def _to_gray(image, invert):
    if image.ndim == 3:
        bgr = image[:, :, :3] if image.shape[2] == 4 else image
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return 255 - gray if invert else gray


@register_node("cv/TextRecognitionCRNN")
class TextRecognitionCRNNNode:
    NAME = "Text Recognition (CRNN)"
    CATEGORY = "OpenCV/OCR"
    INPUTS = [{"name": "image", "type": "IMAGE"}]
    OUTPUTS = [{"name": "text", "type": "TEXT"}]
    WIDGETS = [
        {"name": "invert", "type": "BOOL", "default": False},
    ]

    def run(self, image, invert=False):
        model = _get_crnn_model()
        return {"text": model.recognize(_to_gray(image, invert))}
