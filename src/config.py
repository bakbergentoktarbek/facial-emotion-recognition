# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fer2013")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

IMG_SIZE = 48  # FER-2013 суреттерінің өлшемі

# Папкалардың алфавиттік реті = модель шығысының реті. Бұл ретті ӨЗГЕРТПЕҢІЗ!
EMOTIONS_EN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTIONS_KZ = {
    "angry": "Ашу", "disgust": "Жиреніш", "fear": "Қорқыныш", "happy": "Қуаныш",
    "neutral": "Бейтарап", "sad": "Мұң", "surprise": "Таңдану",
}
# Рамка түстері (RGB)
EMOTION_COLORS = {
    "angry": (231, 76, 60), "disgust": (142, 68, 173), "fear": (52, 73, 94),
    "happy": (46, 204, 113), "neutral": (127, 140, 141), "sad": (52, 152, 219),
    "surprise": (243, 156, 18),
}

CNN_PATH = os.path.join(MODELS_DIR, "emotion_cnn.keras")
MOBILENET_PATH = os.path.join(MODELS_DIR, "emotion_mobilenet.keras")
KDEF_CNN_PATH = os.path.join(MODELS_DIR, "emotion_cnn_kdef.keras")
KDEF_MOBILENET_PATH = os.path.join(MODELS_DIR, "emotion_mobilenet_kdef.keras")
MODEL_PATH = CNN_PATH  # FER бағалауының әдепкі моделі
WEBCAM_MODEL_PATH = KDEF_MOBILENET_PATH
MODEL_ALIASES = {
    "mobilenet-kdef": KDEF_MOBILENET_PATH,
    "cnn-kdef": KDEF_CNN_PATH,
    "cnn-fer": CNN_PATH,
    "mobilenet-fer": MOBILENET_PATH,
}

# Қазақ әріптерін (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, І) жаза алатын қаріптер — біріншісі табылғаны алынады
FONT_CANDIDATES = [
    os.path.join(BASE_DIR, "fonts", "DejaVuSans.ttf"),
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
