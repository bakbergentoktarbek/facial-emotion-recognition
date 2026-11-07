"""Verify camera inference for both KDEF models and both legacy FER models.

No camera is opened, no model is trained, and no images are saved.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import tensorflow as tf
from config import MODEL_ALIASES, KDEF_MOBILENET_PATH, WEBCAM_MODEL_PATH
from data_kdef import prepare_image
from predictor import EmotionPredictor
from webcam_demo import annotate


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class OneFace:
    def detect(self, frame):
        return [(20, 18, 70, 80)]


def main():
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    assert WEBCAM_MODEL_PATH == KDEF_MOBILENET_PATH
    hashes = {path: digest(path) for path in MODEL_ALIASES.values()}
    y, x = np.indices((123, 137))
    image = np.stack((x * 3 % 256, y * 5 % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    shapes = {"mobilenet-kdef": (1, 160, 160, 3), "cnn-kdef": (1, 96, 96, 1),
              "cnn-fer": (1, 48, 48, 1), "mobilenet-fer": (1, 48, 48, 1)}
    for alias, shape in shapes.items():
        tf.keras.backend.clear_session()
        predictor = EmotionPredictor(alias)
        batch = predictor.preprocess(image)
        assert batch.shape == shape and batch.dtype == np.float32
        assert batch.min() >= 0 and batch.max() <= 255 and batch.max() > 1
        if predictor.is_kdef:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            training_input = prepare_image(rgb, predictor.height, predictor.channels).numpy()
            np.testing.assert_array_equal(batch[0], training_input)
            # Fixed crop removes known top/bottom borders, not the middle.
            bordered = np.full((100, 100, 3), 200, np.uint8)
            bordered[:8] = 0
            bordered[-8:] = 0
            np.testing.assert_allclose(predictor.preprocess(bordered), 200, atol=1)
        else:
            legacy = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(batch[0, ..., 0], legacy)
        if predictor.channels == 3:
            red_bgr = np.zeros((32, 32, 3), np.uint8)
            red_bgr[..., 2] = 255
            rgb_batch = predictor.preprocess(red_bgr)
            np.testing.assert_array_equal(rgb_batch[..., 0], 255)
            np.testing.assert_array_equal(rgb_batch[..., 1:], 0)
        emotion, confidence, probabilities = predictor.predict(image)
        assert probabilities.shape == (7,) and np.isfinite(probabilities).all()
        assert 0 <= confidence <= 1
        np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-5)
        for invalid in (None, np.empty((0, 0, 3), np.uint8), np.zeros((4, 4), np.uint8)):
            try:
                predictor.preprocess(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid face image was accepted")
        frame, ema = annotate(image.copy(), OneFace(), predictor)
        assert frame.shape == image.shape and frame.dtype == np.uint8 and ema.shape == (7,)
        _, next_ema = annotate(image.copy(), OneFace(), predictor, ema)
        np.testing.assert_allclose(next_ema.sum(), 1, atol=1e-5)
        print(f"OK {alias}: {predictor.display_name}; input {shape[1:]}; seven probabilities; annotation/EMA", flush=True)
    assert all(digest(path) == value for path, value in hashes.items())
    for report in (ROOT / "reports/comparison_cnn_mobilenet.json", ROOT / "reports/kdef/comparison.json"):
        for result in json.loads(report.read_text(encoding="utf-8")).values():
            assert digest(ROOT / "models" / result["model"]) == result["model_sha256"]
    print("All four checkpoints unchanged; no camera or test dataset used.", flush=True)


if __name__ == "__main__":
    main()
