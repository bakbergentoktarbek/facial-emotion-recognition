# src/predictor.py
import functools
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import EMOTIONS_EN, EMOTIONS_KZ, EMOTION_COLORS, FONT_CANDIDATES, IMG_SIZE, MODEL_PATH


class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train CNN first; see README.md.")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), np.float32), training=False)  # "қыздыру"

    @staticmethod
    def preprocess(face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return gray.astype("float32")[None, ..., None]  # (1, 48, 48, 1): batch + channel өлшемдері

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
