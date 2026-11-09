"""Per-browser game state; reuse the desktop game's rounds and scoring rules."""
import math
import time

import numpy as np
from config import EMOTIONS_EN, EMOTIONS_KZ
from game import EmotionGame, EMA_ALPHA, PREP_TIME, PLAY_TIME, ROUNDS


class WebGameSession(EmotionGame):
    def __init__(self):
        # No camera, model or detector here: Gradio deep-copies session state.
        # Inference uses the web application's shared, warmed model cache.
        self.best_score = 0
        self.model_alias = None
        self.last_frame_at = None
        self.camera_frame_at = None
        self.current_probability = 0.0
        self.run_id = 0
        self.reset()

    def save_highscore(self):
        # The browser session keeps its own record, without a shared JSON file.
        pass

    def begin(self, alias, now=None):
        now = time.perf_counter() if now is None else now
        self.reset()
        self.run_id += 1
        self.model_alias = alias
        self.last_frame_at = now
        self.current_probability = 0.0
        self.next_round(now)

    def stop(self):
        self.reset()
        self.run_id += 1
        self.last_frame_at = None
        self.current_probability = 0.0

    @property
    def active(self):
        return self.state in ("prep", "play", "result")

    def consume(self, scores, now=None):
        now = time.perf_counter() if now is None else now
        if not self.active:
            return
        dt = max(0.0, now - self.last_frame_at) if self.last_frame_at is not None else 0.0
        self.last_frame_at = now
        # No frames means no observations. Pause the round over long gaps,
        # and never award a hold bonus for a disconnected/stalled camera.
        if dt > 1.0:
            self.t0 += dt
        if dt > 0.5:
            self.held, self.ema = 0.0, None
        if scores:
            probs = np.array([scores[EMOTIONS_KZ[e]] for e in EMOTIONS_EN], dtype=np.float32)
            self.ema = probs if self.ema is None else EMA_ALPHA * probs + (1 - EMA_ALPHA) * self.ema
            top = EMOTIONS_EN[int(self.ema.argmax())]
            self.current_probability = float(self.ema[EMOTIONS_EN.index(self.target)])
        else:
            self.ema = None
            top, self.current_probability = None, 0.0
        self.update(now, dt, self.current_probability, top)

    def summary(self, model_name="", now=None):
        now = time.perf_counter() if now is None else now
        footer = f"\n\n**Ұпай: {self.total}/{ROUNDS * 120} · Осы сессиядағы рекорд: {self.best_score}**"
        if model_name:
            footer += f"\n\nМодель: {model_name}. Модель жаңа ойын басталғанда таңдалады."
        if self.state == "menu":
            return "## Эмоцияны қайтала!\n\nКамераны қосып, «Ойынды бастау / қайта ойнау» басыңыз." + footer
        if self.state == "over":
            rounds = ", ".join(str(score) for score in self.scores)
            return f"## Ойын аяқталды!\n\nРаунд ұпайлары: {rounds}.\n\nҚайта ойнауға болады." + footer
        title = f"## Раунд {self.round}/{ROUNDS} · {EMOTIONS_KZ[self.target].upper()}"
        if self.state == "prep":
            body = f"Дайындал... **{max(0, math.ceil(PREP_TIME - (now - self.t0)))} с**"
        elif self.state == "play":
            left = max(0.0, PLAY_TIME - (now - self.t0))
            body = (f"Эмоцияны көрсетіңіз! **{left:.1f} с**\n\n"
                    f"Мақсат ықтималдығы: **{self.current_probability * 100:.0f}%** · "
                    f"Ең жақсысы: {self.best_prob * 100:.0f}%\n\n"
                    + ("Бонус +20 алынды!" if self.bonus_earned else "Бонус үшін эмоцияны 1 секунд үздіксіз ұстаңыз."))
        else:
            body = f"**+{self.scores[-1]} ұпай**" + (" · бонус +20!" if self.bonus_earned else "")
        return title + "\n\n" + body + footer
