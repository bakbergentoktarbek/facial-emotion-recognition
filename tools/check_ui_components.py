"""Ensure the redesign reflects existing predictions and game state faithfully."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import EMOTIONS_EN, EMOTIONS_KZ
from ui_components import HINTS, game_panels, result_panel
from web_game import WebGameSession


def main():
    initial = result_panel()
    assert "Нәтиже осы жерде көрсетіледі" in initial and "emotion-row" not in initial
    missing = result_panel({}, analyzed=True)
    assert "Бет анықталмады" in missing and "emotion-row" not in missing and "%" not in missing
    scores = {EMOTIONS_KZ[e]: .05 for e in EMOTIONS_EN}
    scores[EMOTIONS_KZ["happy"]] = .7
    html = result_panel(scores, True, "CNN")
    assert html.count('class="emotion-row') == 7 and html.count('is-leading') == 1
    assert '<h2>Қуаныш</h2>' in html and '70<small>%</small>' in html
    assert "Бет анықталмады" not in html

    state = WebGameSession()
    state.begin("cnn-kdef", 0)
    state.target = "surprise"
    state.start_play(2)
    before = copy.deepcopy(state.__dict__)
    stats, target, progress = game_panels(state, "CNN", now=5.5)
    assert state.__dict__ == before
    assert 'urgent' in stats and '1.5' in stats and HINTS["surprise"] in target
    assert 'Модель бағасы' in progress and 'Бет әлпетін сақтаңыз' in progress
    assert 'aria-valuenow="0.0"' in progress
    state.current_probability = .81
    assert game_panels(state, now=5.5)[1] == target  # Hint is task-based, not score-based.
    assert 'aria-valuenow="81.0"' in game_panels(state, now=5.5)[2]
    assert 'urgent' not in game_panels(state, now=3)[0]
    state.stop()
    assert 'aria-valuenow="0.0"' in game_panels(state)[2]
    print("UI checks passed: empty/no-face, seven bars, one leader, state-based timer, hints and scores.")


if __name__ == "__main__":
    main()
