# src/model_cnn.py
from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SIZE


def augmentation():
    """Тек model.fit() кезінде жұмыс істейді; predict/демо кезінде автоматты түрде өшеді."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),          # ±18°
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="augmentation")


def conv_block(x, filters, dropout):
    # Екі Conv қатарынан: 3×3 + 3×3 ≈ 5×5 көру аймағы, бірақ параметр аз
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)  # bias-ты BN алмастырады
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_cnn(num_classes=7):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)       # нормализация модель ІШІНДЕ
    x = conv_block(x, 32, 0.25)              # 48 → 24
    x = conv_block(x, 64, 0.25)              # 24 → 12
    x = conv_block(x, 128, 0.30)             # 12 → 6
    x = conv_block(x, 256, 0.30)             # 6 → 3
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="emotion_cnn")


if __name__ == "__main__":
    build_cnn().summary()  # ≈ 1.2 млн параметр
