# src/webcam_demo.py
import argparse
import time
import cv2
from config import EMOTIONS_EN, MODEL_PATH
from face_detect import get_detector
from predictor import EmotionPredictor, crop_face, draw_label, put_text

EMA_ALPHA = 0.4


def annotate(frame, detector, predictor, ema_state=None):
    """Кадрдағы барлық бетті белгілейді. ema_state — ең үлкен бет үшін тегістеу."""
    boxes = detector.detect(frame)
    if not boxes:
        return put_text(frame, "Бет табылмады", (10, 10), 26, bg=(192, 57, 43)), None
    biggest = max(boxes, key=lambda b: b[2] * b[3])
    for box in boxes:
        face = crop_face(frame, box)
        if face.size == 0:
            continue
        _, _, probs = predictor.predict(face)
        if box == biggest and ema_state is not None:
            probs = EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_state
        if box == biggest:
            ema_state = probs
        i = int(probs.argmax())
        frame = draw_label(frame, box, EMOTIONS_EN[i], float(probs[i]))
    return frame, ema_state


def run_image(path, detector, predictor):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Сурет оқылмады: {path}")
    frame, _ = annotate(frame, detector, predictor)
    cv2.imshow("Emotion", frame)
    cv2.waitKey(0)


def run_webcam(cam_index, detector, predictor):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Камера ашылмады: --camera 1 көріңіз немесе Zoom/Teams-ті жабыңыз")
    ema, prev = None, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame, ema = annotate(frame, detector, predictor, ema)
        now = time.time()
        fps, prev = 1.0 / max(now - prev, 1e-6), now
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Emotion recognition (q - exit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--detector", choices=["mediapipe", "haar"], default="mediapipe")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", help="Камера орнына бір суретті талдау")
    args = p.parse_args()
    detector, predictor = get_detector(args.detector), EmotionPredictor(args.model)
    if args.image:
        run_image(args.image, detector, predictor)
    else:
        run_webcam(args.camera, detector, predictor)
