# src/game.py
import argparse
import json
import math
import os
import random
import time
import cv2
from camera import open_camera, read_camera_frame
from config import BASE_DIR, EMOTIONS_EN, EMOTIONS_KZ, EMOTION_COLORS, WEBCAM_MODEL_PATH
from face_detect import get_detector
from predictor import EmotionPredictor, crop_face, draw_label, put_text

ROUNDS = 5
PREP_TIME, PLAY_TIME, RESULT_TIME = 2.0, 5.0, 2.5
GAME_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "neutral"]  # жиреніш тым қиын — алынбаған
EMA_ALPHA = 0.4
HOLD_FOR_BONUS = 1.0  # секунд
HS_FILE = os.path.join(BASE_DIR, "highscore.json")


class EmotionGame:
    def __init__(self, model_path=WEBCAM_MODEL_PATH, detector_name="mediapipe", highscore_path=HS_FILE):
        self.detector = get_detector(detector_name)
        self.predictor = EmotionPredictor(model_path)
        self.highscore_path = highscore_path
        self.best_score = self.load_highscore()
        self.reset()

    # ---------- рекорд ----------
    def load_highscore(self):
        try:
            with open(self.highscore_path, encoding="utf-8") as f:
                value = json.load(f)["best"]
                return value if type(value) is int and 0 <= value <= ROUNDS * 120 else 0
        except (OSError, KeyError, ValueError, TypeError):
            return 0

    def save_highscore(self):
        with open(self.highscore_path, "w", encoding="utf-8") as f:
            json.dump({"best": self.best_score}, f)

    # ---------- күйлер ----------
    def reset(self):
        self.state, self.round, self.total, self.scores = "menu", 0, 0, []
        self.target, self.ema = None, None

    def next_round(self, now=None):
        self.round += 1
        options = [e for e in GAME_EMOTIONS if e != self.target]  # қатарынан қайталанбасын
        self.target = random.choice(options)
        self.ema = None
        self.state, self.t0 = "prep", time.perf_counter() if now is None else now

    def start_play(self, now=None):
        self.state, self.t0 = "play", time.perf_counter() if now is None else now
        self.best_prob, self.held = 0.0, 0.0
        self.bonus_earned, self.ema = False, None

    def finish_round(self, now=None):
        points = int(round(self.best_prob * 100)) + (20 if self.bonus_earned else 0)
        self.scores.append(points)
        self.total += points
        self.state, self.t0 = "result", time.perf_counter() if now is None else now

    # ---------- модель ----------
    def analyze(self, frame):
        """Ең үлкен бетті талдап, (кадр, мақсат ықтималдығы, №1 эмоция) қайтарады."""
        boxes = self.detector.detect(frame)
        if not boxes:
            self.ema = None
            return put_text(frame, "Бетіңіз көрінбейді!", (10, 60), 26, bg=(192, 57, 43)), 0.0, None
        box = max(boxes, key=lambda b: b[2] * b[3])
        face = crop_face(frame, box)
        if face.size == 0:
            self.ema = None
            return frame, 0.0, None
        _, _, probs = self.predictor.predict(face)
        self.ema = probs if self.ema is None else EMA_ALPHA * probs + (1 - EMA_ALPHA) * self.ema
        top = EMOTIONS_EN[int(self.ema.argmax())]
        frame = draw_label(frame, box, top, float(self.ema.max()))
        target_prob = float(self.ema[EMOTIONS_EN.index(self.target)]) if self.target else 0.0
        return frame, target_prob, top

    # ---------- сурет салу ----------
    @staticmethod
    def draw_bar(frame, value, best, y=110):
        """Мақсатты эмоция ықтималдығының жолағы + ең жақсы нәтиже белгісі."""
        x0, w, h = 20, 300, 22
        cv2.rectangle(frame, (x0, y), (x0 + w, y + h), (80, 80, 80), -1)
        cv2.rectangle(frame, (x0, y), (x0 + int(w * value), y + h), (113, 204, 46), -1)
        bx = x0 + int(w * best)
        cv2.line(frame, (bx, y - 4), (bx, y + h + 4), (0, 215, 255), 3)
        return frame

    def draw_hud(self, frame, now, target_prob):
        W = frame.shape[1]
        if self.state == "menu":
            frame = put_text(frame, "«Эмоцияны қайтала!»", (20, 20), 40, bg=(44, 62, 80))
            frame = put_text(frame, "SPACE — бастау   Q — шығу", (20, 85), 26, bg=(44, 62, 80))
            frame = put_text(frame, f"Рекорд: {self.best_score}", (20, 130), 26, bg=(44, 62, 80))
        elif self.state in ("prep", "play"):
            name = EMOTIONS_KZ[self.target].upper()
            frame = put_text(frame, f"Раунд {self.round}/{ROUNDS}:  {name}", (20, 15), 34,
                             bg=EMOTION_COLORS[self.target])
            if self.state == "prep":
                left = math.ceil(PREP_TIME - (now - self.t0))
                frame = put_text(frame, f"Дайындал... {left}", (20, 70), 28, bg=(44, 62, 80))
            else:
                left = max(0.0, PLAY_TIME - (now - self.t0))
                frame = put_text(frame, f"{left:.1f} с", (20, 65), 28, bg=(44, 62, 80))
                frame = self.draw_bar(frame, target_prob, self.best_prob)
            frame = put_text(frame, f"Ұпай: {self.total}", (W - 170, 15), 28, bg=(44, 62, 80))
        elif self.state == "result":
            pts = self.scores[-1]
            bonus = " (бонус +20!)" if self.bonus_earned else ""
            frame = put_text(frame, f"+{pts} ұпай{bonus}", (20, 20), 38, bg=(39, 174, 96))
        elif self.state == "over":
            ratio = self.total / (ROUNDS * 120)
            verdict = "Нағыз актер!" if ratio >= 0.8 else "Жақсы нәтиже!" if ratio >= 0.5 else "Тағы жаттығайық :)"
            frame = put_text(frame, f"Ойын аяқталды! {self.total}/{ROUNDS * 120}", (20, 20), 34, bg=(44, 62, 80))
            frame = put_text(frame, verdict + f"   Рекорд: {self.best_score}", (20, 75), 28, bg=(44, 62, 80))
            frame = put_text(frame, "Раундтар: " + ", ".join(map(str, self.scores)), (20, 120), 24, bg=(44, 62, 80))
            frame = put_text(frame, "SPACE — қайта ойнау   Q — шығу", (20, 160), 24, bg=(44, 62, 80))
        return put_text(frame, self.predictor.display_name + " | Q / Esc — шығу",
                        (10, frame.shape[0] - 30), 20, bg=(44, 62, 80))

    # ---------- негізгі цикл ----------
    def update(self, now, dt, target_prob, top):
        if self.state == "prep" and now - self.t0 >= PREP_TIME:
            self.start_play(now)
        elif self.state == "play":
            if now - self.t0 <= PLAY_TIME:
                self.best_prob = max(self.best_prob, target_prob)
                # Bonus requires a continuous second, not scattered matches.
                # A stalled camera is not evidence of a held expression.
                if top == self.target and 0 <= dt <= 0.5:
                    self.held += min(dt, max(0.0, now - self.t0))
                    if self.held >= HOLD_FOR_BONUS:
                        self.bonus_earned = True
                else:
                    self.held = 0.0
            if now - self.t0 >= PLAY_TIME:
                self.finish_round(now)
        elif self.state == "result" and now - self.t0 >= RESULT_TIME:
            if self.round >= ROUNDS:
                self.state = "over"
                if self.total > self.best_score:
                    self.best_score = self.total
                    self.save_highscore()
            else:
                self.next_round(now)

    def run(self, cam_index=0, backend="auto"):
        cap = open_camera(cam_index, backend)
        title = "Emotion game"
        prev = time.perf_counter()
        try:
            while True:
                frame = cv2.flip(read_camera_frame(cap), 1)
                frame, target_prob, top = self.analyze(frame)
                now = time.perf_counter()
                dt, prev = now - prev, now
                self.update(now, dt, target_prob, top)
                frame = self.draw_hud(frame, now, target_prob)
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" ") and self.state in ("menu", "over"):
                    self.reset()
                    self.next_round(now)
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat the expression; Space starts, Q/Esc exits.")
    parser.add_argument("--model", default=WEBCAM_MODEL_PATH,
                        help="mobilenet-kdef (default), cnn-kdef, cnn-fer, mobilenet-fer, or .keras path")
    parser.add_argument("--detector", choices=["mediapipe", "haar"], default="mediapipe")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    args = parser.parse_args()
    print("Loading game detector and emotion model...", flush=True)
    game = EmotionGame(args.model, args.detector)
    print(f"Game ready: {game.predictor.display_name}. Space starts; Q/Esc exits.", flush=True)
    game.run(args.camera, args.backend)
