# src/predictor.py
import functools
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import EMOTIONS_EN, EMOTIONS_KZ, EMOTION_COLORS, FONT_CANDIDATES, MODEL_PATH, MODEL_ALIASES


class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(MODEL_ALIASES.get(str(model_path), model_path))
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Download the matching .keras file "
                "from the project's GitHub Releases into models/; see README.md.")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        shape = self.model.input_shape
        if (not isinstance(shape, tuple) or len(shape) != 4 or
                any(not isinstance(value, int) or value < 1 for value in shape[1:]) or
                shape[-1] not in (1, 3) or self.model.output_shape != (None, len(EMOTIONS_EN))):
            raise ValueError("Expected one image input with 1 or 3 channels and seven emotion outputs")
        self.height, self.width, self.channels = shape[1:]
        self.is_kdef = self.model.name in ("emotion_cnn_kdef", "emotion_mobilenet_kdef")
        if self.is_kdef and self.height != self.width:
            raise ValueError("KDEF preprocessing requires a square model input")
        self.display_name = {
            "emotion_cnn_kdef": "CNN (KDEF)",
            "emotion_mobilenet_kdef": "MobileNetV2 (KDEF)",
            "emotion_cnn": "CNN (FER-2013)",
            "emotion_mobilenetv2": "MobileNetV2 (FER-2013)",
        }.get(self.model.name, self.model_path.stem)
        self.model(np.zeros((1, self.height, self.width, self.channels), np.float32), training=False)

    def preprocess(self, face_bgr):
        if (not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0 or
                face_bgr.ndim != 3 or face_bgr.shape[-1] != 3):
            raise ValueError("Expected a nonempty BGR face image with three channels")
        if self.is_kdef:
            # TensorFlow crop/grayscale/resize/round must match the training
            # transform. OpenCV supplies BGR; KDEF was decoded as RGB.
            from data_kdef import prepare_image
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            image = prepare_image(rgb, self.height, self.channels).numpy()
        else:
            # Preserve the exact preprocessing used by the original FER demo.
            conversion = cv2.COLOR_BGR2GRAY if self.channels == 1 else cv2.COLOR_BGR2RGB
            image = cv2.cvtColor(face_bgr, conversion)
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if self.channels == 1:
                image = image[..., None]
        # Scaling is already inside every saved project model.
        return image.astype("float32")[None, ...]

    def predict(self, face_bgr):
        # model(...) бір сурет үшін model.predict(...)-тен бірнеше есе жылдам
        probs = self.model(self.preprocess(face_bgr), training=False).numpy()[0]
        i = int(np.argmax(probs))
        return EMOTIONS_EN[i], float(probs[i]), probs


def crop_face(frame, box, margin=0.08):
    """Рамканы сәл кеңейтіп кесеміз: FER суреттерінде маңдай мен иек те бар."""
    x, y, w, h = box
    H, W = frame.shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    return frame[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]


def label_text(emotion, conf):
    return f"{EMOTIONS_KZ[emotion]} — {conf * 100:.0f}%"


@functools.lru_cache(maxsize=8)
def get_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    print("⚠ Қазақша қаріп табылмады: fonts/DejaVuSans.ttf салыңыз")
    return ImageFont.load_default()


def put_text(frame_bgr, text, xy, size=24, color=(255, 255, 255), bg=None):
    """Кадрға Unicode мәтін жазады. Түстер RGB форматында (PIL), ал кадр BGR (OpenCV)."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = get_font(size)
    if bg is not None:
        l, t, r, b = draw.textbbox(xy, text, font=font)
        draw.rectangle([l - 6, t - 4, r + 6, b + 4], fill=bg)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_label(frame_bgr, box, emotion, conf):
    x, y, w, h = box
    color = EMOTION_COLORS[emotion]
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color[::-1], 3)  # RGB → BGR
    return put_text(frame_bgr, label_text(emotion, conf), (x + 4, max(4, y - 34)), 24, bg=color)
