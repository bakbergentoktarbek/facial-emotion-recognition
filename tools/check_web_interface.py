"""Check the web callbacks and local HTTP API using synthetic images only."""
import hashlib
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(ROOT / "src"))

import cv2
import httpx
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file
import app_gradio as web
from config import EMOTIONS_EN, EMOTIONS_KZ, MODEL_ALIASES
from predictor import crop_face


class Faces:
    def detect(self, frame):
        # Deliberately unsorted and overlapping: drawing a label must not
        # contaminate the other face's classifier input.
        return [(20, 10, 100, 110), (60, 55, 35, 40)]


def main():
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in MODEL_ALIASES.values()}
    y, x = np.indices((180, 200))
    image = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    engine = web.WebRecognizer()
    assert engine.analyze(None)[:2] == (None, {})
    assert engine.detector is None and not engine.predictors
    for invalid in (np.zeros((8, 8), np.uint8), np.empty((0, 0, 3), np.uint8)):
        try:
            engine.analyze(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed image was accepted")
    try:
        engine.analyze(image, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown alias was accepted")

    # Exercise actual face detector initialization and the no-face result.
    blank, scores, status = engine.analyze(np.zeros_like(image))
    assert blank.shape == image.shape and scores == {} and not engine.predictors
    engine.detector = Faces()
    for alias in MODEL_ALIASES:
        annotated, scores, status = engine.analyze(image, alias)
        predictor = engine.predictors[alias]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        expected = predictor.predict(crop_face(bgr, Faces().detect(bgr)[0]))[2]
        np.testing.assert_allclose([scores[EMOTIONS_KZ[e]] for e in EMOTIONS_EN], expected,
                                   atol=1e-6)
        assert annotated.shape == image.shape and web.MODEL_NAMES[alias] in status
        assert "2." in status
        before = engine.predictors[alias]
        engine.analyze(image, alias)
        assert engine.predictors[alias] is before
        print(f"OK web: {alias}; largest-face scores; cached model; overlapping faces", flush=True)

    web.recognizer = engine
    stream_image, scores, _ = web.analyze_stream(np.tile(image, (4, 4, 1)), "cnn-kdef")
    assert max(stream_image.shape[:2]) == 640 and len(scores) == 7
    with patch.object(engine, "analyze", side_effect=FileNotFoundError("synthetic")):
        try:
            web.analyze(image)
        except web.gr.Error as error:
            assert "GitHub Releases" in str(error)
        else:
            raise AssertionError("Missing model must produce a user-facing error")

    cache = ROOT / "reports" / "web_checks"
    cache.mkdir(parents=True, exist_ok=True)
    sample = cache / "synthetic.png"
    Image.fromarray(image).save(sample)
    demo = web.build_demo()
    try:
        _, url, _ = demo.launch(server_name="127.0.0.1", share=False,
                                prevent_thread_lock=True, quiet=True, max_file_size="10mb")
        client = Client(url, verbose=False, download_files=str(cache))
        for endpoint, alias in (("/analyze_photo", "mobilenet-kdef"),
                                ("/analyze_photo", "cnn-fer")):
            result = client.predict(handle_file(str(sample)), alias, api_name=endpoint)
            assert Path(result[0]).is_file()
            assert len(result[1]["confidences"]) == 7
            assert web.MODEL_NAMES[alias] in result[2]
            print(f"OK HTTP {endpoint}: {alias}", flush=True)
        # gradio_client 1.14 does not recognize the browser stream's
        # process_streaming SSE event. Exercise its frame endpoint directly.
        with sample.open("rb") as stream:
            upload = httpx.post(url + "gradio_api/upload",
                                files={"files": (sample.name, stream, "image/png")}, timeout=30)
        upload.raise_for_status()
        response = httpx.post(url + "gradio_api/run/analyze_camera", json={
            "data": [{"path": upload.json()[0], "meta": {"_type": "gradio.FileData"}}, "cnn-kdef"]
        }, timeout=30)
        response.raise_for_status()
        result = response.json()["data"]
        assert len(result[1]["confidences"]) == 7 and "CNN · KDEF" in result[2]
        print("OK HTTP /analyze_camera: frame preprocessing and seven scores", flush=True)
        result = client.predict(None, "cnn-fer", api_name="/analyze_photo")
        assert result[0] is None
    finally:
        demo.close()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    print("Web checks passed; checkpoints unchanged. Physical/browser camera not tested.", flush=True)


if __name__ == "__main__":
    main()
