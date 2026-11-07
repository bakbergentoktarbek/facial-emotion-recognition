"""KDEF-specific models; the FER-2013 architectures and checkpoints stay intact."""
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

from config import BASE_DIR, CNN_PATH
from model_cnn import conv_block

CNN_SIZE = 96
MOBILENET_SIZE = 160
IMAGENET_WEIGHTS = Path(BASE_DIR) / "reports/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"


def augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.025),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.04, 0.04),
        layers.RandomContrast(0.1),
    ], name="kdef_augmentation")


def build_cnn():
    inputs = keras.Input((CNN_SIZE, CNN_SIZE, 1))
    x = augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters, dropout in ((32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)):
        x = conv_block(x, filters, dropout)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(7, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="emotion_cnn_kdef")
    source = keras.models.load_model(CNN_PATH, compile=False)
    # Same weighted layers; spatial resolution/dropout/augmentation can change.
    model.set_weights(source.get_weights())
    return model


def build_mobilenet():
    if not IMAGENET_WEIGHTS.is_file():
        raise FileNotFoundError(f"Cached original ImageNet weights required: {IMAGENET_WEIGHTS}")
    inputs = keras.Input((MOBILENET_SIZE, MOBILENET_SIZE, 3))
    x = augmentation()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    # The convolutional backbone's weights do not depend on input width/height.
    # Reuse the official cached ImageNet 96-resolution variant at 160x160.
    base = keras.applications.MobileNetV2(
        input_shape=(MOBILENET_SIZE, MOBILENET_SIZE, 3),
        include_top=False, weights=str(IMAGENET_WEIGHTS))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(7, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="emotion_mobilenet_kdef"), base


def unfreeze_mobilenet(base, top_layers=100):
    base.trainable = True
    boundary = len(base.layers) - top_layers
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= boundary and not isinstance(layer, layers.BatchNormalization)
