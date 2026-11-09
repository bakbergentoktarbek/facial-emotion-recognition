"""Exercise browser-game state and real Gradio session isolation without a camera."""
import copy
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import httpx
import numpy as np
from PIL import Image
from gradio_client import Client
import app_gradio as web
from config import EMOTIONS_EN, EMOTIONS_KZ
from web_game import WebGameSession


def scores_for(emotion):
    return {EMOTIONS_KZ[e]: float(e == emotion) for e in EMOTIONS_EN}


def main():
    first, second = WebGameSession(), WebGameSession()
    assert copy.deepcopy(first).state == "menu"
    first.begin("cnn-kdef", 0)
    first.consume({}, .25)
    first.consume({}, 10)  # Lost frames pause the preparation instead of skipping it.
    assert first.state == "prep" and first.t0 == 9.75 and second.state == "menu"
    first.consume({}, 10.75)
    first.consume({}, 11.75)
    assert first.state == "play"
    now = 11.75
    for number in range(1, 6):
        assert first.round == number and first.model_alias == "cnn-kdef"
        for _ in range(20):
            now += .25
            first.consume(scores_for(first.target), now)
        assert first.state == "result" and first.scores[-1] == 120
        assert "120" in first.summary(now=now)
        for _ in range(10):
            now += .25
            first.consume({}, now)
        if number < 5:
            assert first.state == "prep"
            for _ in range(8):
                now += .25
                first.consume({}, now)
            assert first.state == "play"
    assert first.state == "over" and first.total == first.best_score == 600
    assert second.total == second.best_score == 0
    first.begin("mobilenet-fer", now)
    assert first.total == 0 and first.best_score == 600 and not first.scores
    first.stop()
    assert not first.active and first.best_score == 600
    print("OK five rounds, 600 points, replay, pause and independent records", flush=True)

    image = np.zeros((180, 200, 3), np.uint8)
    try:
        web.start_web_game("cnn-kdef", second)
    except web.gr.Error:
        pass
    else:
        raise AssertionError("Starting without a camera frame must be rejected")
    with patch.object(web.recognizer, "warm_up") as warm:
        web.analyze_game_frame(image, second)
        board, second = web.start_web_game("cnn-kdef", second)
        warm.assert_called_once_with("cnn-kdef")
    with patch.object(web, "_analyze", side_effect=lambda *args: (second.stop() or image, {}, "")):
        _, _, second = web.analyze_game_frame(image, second)
    assert second.state == "menu" and second.total == 0
    print("OK stopped game ignores an in-flight prediction", flush=True)

    # The UI's state is kept server-side per session, not supplied by a client.
    cache = ROOT / "reports" / "web_game_checks"
    cache.mkdir(parents=True, exist_ok=True)
    sample = cache / "synthetic.png"
    Image.fromarray(image).save(sample)
    demo = web.build_demo()
    with patch.object(web.recognizer, "warm_up"), \
            patch.object(web, "_analyze", return_value=(image, {}, "synthetic")):
        try:
            _, url, _ = demo.launch(server_name="127.0.0.1", share=False,
                                    prevent_thread_lock=True, quiet=True,
                                    app_kwargs={"lifespan": web.cache_lifespan})
            a, b = Client(url, verbose=False), Client(url, verbose=False)
            with sample.open("rb") as stream:
                uploaded = httpx.post(url + "gradio_api/upload",
                                      files={"files": (sample.name, stream, "image/png")}, timeout=30)
            uploaded.raise_for_status()
            ready = httpx.post(url + "gradio_api/run/game_frame", json={
                "data": [{"path": uploaded.json()[0], "meta": {"_type": "gradio.FileData"}}, None],
                "session_hash": a.session_hash,
            }, timeout=30)
            ready.raise_for_status()
            board = a.predict("cnn-kdef", api_name="/start_game")
            board = board[0] if isinstance(board, (tuple, list)) else board
            assert "Раунд 1/5" in board and "CNN · KDEF" in board
            for client, expected in ((a, "Раунд 1/5"), (b, "Эмоцияны қайтала!")):
                response = httpx.post(url + "gradio_api/run/game_frame", json={
                    "data": [{"path": uploaded.json()[0], "meta": {"_type": "gradio.FileData"}}, None],
                    "session_hash": client.session_hash,
                }, timeout=30)
                response.raise_for_status()
                assert expected in response.json()["data"][1]
            board = a.predict(api_name="/stop_game")
            board = board[0] if isinstance(board, (tuple, list)) else board
            assert "Эмоцияны қайтала!" in board and "Раунд 1/5" not in board
            print("OK HTTP start/frame/stop and separate browser session states", flush=True)
        finally:
            demo.close()
    print("Web game checks passed. Synthetic inputs only; no model or dataset changed.", flush=True)


if __name__ == "__main__":
    main()
