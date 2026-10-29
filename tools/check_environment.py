"""Check imports and data layout without opening the webcam or reading secrets."""
import importlib
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    print(f"Python: {sys.version.split()[0]}\nInterpreter: {sys.executable}")
    failed = False
    for name in ("numpy", "tensorflow", "cv2", "mediapipe", "PIL", "matplotlib",
                 "seaborn", "sklearn", "gradio"):
        try:
            module = importlib.import_module(name)
            print(f"OK {name}: {getattr(module, '__version__', 'installed')}")
        except Exception as error:
            failed = True
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    if not failed:
        import tensorflow as tf
        from face_detect import HaarFaceDetector, MediaPipeFaceDetector
        import numpy as np
        print(f"TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")
        for detector_class in (HaarFaceDetector, MediaPipeFaceDetector):
            try:
                detector = detector_class()
                boxes = detector.detect(np.zeros((128, 128, 3), dtype=np.uint8))
                assert boxes == [], f"Unexpected boxes: {boxes}"
                if hasattr(detector, "detector"):
                    detector.detector.close()
                print(f"OK {detector_class.__name__}: blank frame")
            except Exception as error:
                failed = True
                print(f"FAIL {detector_class.__name__}: {error}")
        try:
            from model_cnn import build_cnn
            from predictor import EmotionPredictor
            model = build_cnn()
            with tempfile.TemporaryDirectory(prefix="emotion-check-") as temporary:
                path = Path(temporary) / "smoke.keras"
                model.save(path)
                predictor = EmotionPredictor(path)
                emotion, confidence, probabilities = predictor.predict(
                    np.zeros((64, 64, 3), dtype=np.uint8))
                assert probabilities.shape == (7,)
                assert np.isfinite(probabilities).all()
                assert np.isclose(probabilities.sum(), 1, atol=1e-5)
            print("OK CNN: build, save/reload, preprocessing, 7 probabilities (untrained)")
        except Exception as error:
            failed = True
            print(f"FAIL CNN smoke check: {type(error).__name__}: {error}")
    expected = [ROOT / "data" / "fer2013" / split / label
                for split in ("train", "test")
                for label in ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")]
    print("Dataset folders: " + ("present" if all(p.is_dir() for p in expected)
                                 else "not ready; follow README.md"))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
