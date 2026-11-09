"""Check scoring, all five rounds, model integration and camera cleanup without a camera."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import game as module
from config import MODEL_ALIASES


CHECKS = ROOT / "reports" / "game_checks"
CHECKS.mkdir(parents=True, exist_ok=True)
RECORD = CHECKS / "highscore.json"


def make_game():
    with patch.object(module, "get_detector"), patch.object(module, "EmotionPredictor") as factory:
        factory.return_value.display_name = "Synthetic check"
        game = module.EmotionGame(highscore_path=RECORD)
        factory.assert_called_once_with(module.WEBCAM_MODEL_PATH)
        return game


def main():
    RECORD.write_text('{"best": 0}', encoding="utf-8")
    game = make_game()
    image = np.full((480, 640, 3), 100, np.uint8)

    # Disjoint matches do not earn a continuous-hold bonus; missing faces and
    # long capture gaps reset the hold, and an earned bonus remains earned.
    game.target = "happy"
    game.start_play(0)
    game.update(.4, .4, .8, "happy")
    game.update(.5, .1, .2, "sad")
    game.update(.9, .4, .8, "happy")
    assert not game.bonus_earned and game.held == .4
    game.update(1.0, .1, 0, None)
    assert game.held == 0
    game.update(2, 1, .9, "happy")
    assert game.held == 0 and not game.bonus_earned
    for now in (2.25, 2.5, 2.75, 3):
        game.update(now, .25, .9, "happy")
    assert game.bonus_earned
    game.update(3.1, .1, 0, None)
    assert game.bonus_earned and game.held == 0
    game.update(5.1, 2, 1.0, "happy")
    assert game.state == "result" and game.scores == [110]  # Late prediction excluded.
    print("OK continuous bonus, missing face, capture stall, late frame", flush=True)

    # Run a full five-round game with synthetic time, no sleeping or GUI.
    game.reset()
    game.next_round(10)
    now = 10.0
    previous_target = None
    for number in range(1, module.ROUNDS + 1):
        assert game.round == number and game.target != previous_target
        previous_target = game.target
        game.draw_hud(image.copy(), now, 0)
        now += module.PREP_TIME
        game.update(now, .1, 0, None)
        assert game.state == "play" and game.ema is None
        for _ in range(20):
            now += .25
            game.update(now, .25, .9, game.target)
        assert game.state == "result" and game.scores[-1] == 110
        game.draw_hud(image.copy(), now, .9)
        now += module.RESULT_TIME
        game.update(now, .1, 0, None)
    assert game.state == "over" and game.total == 550
    assert json.loads(RECORD.read_text())["best"] == 550
    game.draw_hud(image.copy(), now, 0)
    game.reset()
    assert game.state == "menu" and game.scores == [] and game.best_score == 550
    print("OK five rounds, HUD, saved record, replay", flush=True)

    # Read failure and every exit mechanism must release the camera.
    for key, visible, failure in ((ord("q"), 1, False), (27, 1, False),
                                   (ord("Q"), 1, False), (0, 0, False), (0, 1, True)):
        cap = Mock()
        with patch.object(module, "open_camera", return_value=cap) as opener, \
                patch.object(module, "read_camera_frame", return_value=image,
                             side_effect=RuntimeError("synthetic frame failure") if failure else None), \
                patch.object(game, "analyze", return_value=(image, 0, None)), \
                patch.object(module.cv2, "imshow"), \
                patch.object(module.cv2, "waitKey", return_value=key), \
                patch.object(module.cv2, "getWindowProperty", return_value=visible), \
                patch.object(module.cv2, "destroyAllWindows") as destroy:
            try:
                game.run(2, "dshow")
            except RuntimeError:
                assert failure
            else:
                assert not failure
            opener.assert_called_once_with(2, "dshow")
            cap.release.assert_called_once()
            destroy.assert_called_once()
    print("OK Q/Esc/close/error all release camera", flush=True)

    # Real saved models, synthetic face region, no physical camera or dataset.
    class Face:
        def detect(self, frame):
            return [(100, 100, 160, 180)]

    for alias in MODEL_ALIASES:
        with patch.object(module, "get_detector", return_value=Face()):
            actual = module.EmotionGame(alias, highscore_path=RECORD)
        actual.target = "happy"
        result, probability, top = actual.analyze(image.copy())
        assert result.shape == image.shape and 0 <= probability <= 1
        assert top in module.EMOTIONS_EN
        print(f"OK game inference: {alias}", flush=True)
    print("Game checks passed. Actual camera and manual play remain to be checked.", flush=True)


if __name__ == "__main__":
    main()
