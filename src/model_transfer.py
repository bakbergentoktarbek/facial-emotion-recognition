# src/model_transfer.py
from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SIZE
from model_cnn import augmentation


def build_mobilenet(num_classes=7, size=96):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))       # демо үшін кіріс бұрынғыдай: 48×48×1
    x = augmentation()(inputs)
    x = layers.Resizing(size, size)(x)                         # 48 → 96
    x = layers.Concatenate()([x, x, x])                        # сұр → 3 арна
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)          # 0..255 → -1..1
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)  # BatchNorm статистикасы fine-tuning кезінде де өзгермейді
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="emotion_mobilenetv2"), base


def unfreeze_top(base, n_layers=30):
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
