# src/face_detect.py
import cv2


class HaarFaceDetector:
    """Классикалық Viola–Jones детекторы. Тез, бірақ тек тура қараған бетті жақсы табады."""

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load Haar Cascade: {path}")

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # Haar түсті қажет етпейді
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # әр қадамда суретті 10%-ға кішірейтіп, әр өлшемдегі бетті іздейді
            minNeighbors=5,    # көп болса — жалған табу аз, бірақ кейбір бет өткізіліп кетеді
            minSize=(60, 60),  # тым кішкентай "беттерді" елемейміз
        )
        return [tuple(int(v) for v in f) for f in faces]  # [(x, y, w, h), ...]


class MediaPipeFaceDetector:
    """Нейрондық желіге негізделген детектор. Сенімдірек, бұрылған бетті де табады."""

    def __init__(self, min_confidence=0.5):
        import mediapipe as mp
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,                    # 0 — камераға жақын (2 м дейін) беттер
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        result = self.detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))  # MediaPipe RGB күтеді
        boxes = []
        for det in result.detections or []:
            b = det.location_data.relative_bounding_box  # 0..1 аралығындағы салыстырмалы координаттар
            x, y = max(0, int(b.xmin * w)), max(0, int(b.ymin * h))
            right = min(w, int((b.xmin + b.width) * w))
            bottom = min(h, int((b.ymin + b.height) * h))
            if right > x and bottom > y:
                boxes.append((x, y, right - x, bottom - y))
        return boxes


def get_detector(name="mediapipe"):
    if name == "mediapipe":
        try:
            return MediaPipeFaceDetector()
        except Exception as e:  # MediaPipe орнатылмаса — Haar-ға ауысамыз
            print(f"MediaPipe қолжетімсіз ({e}), Haar Cascade қолданылады.")
    return HaarFaceDetector()


if __name__ == "__main__":
    # 1-апта тапсырмасы: екі детекторды қатар салыстыру. 'm' — ауыстыру, 'q' — шығу
    import argparse
    from camera import open_camera, read_camera_frame
    parser = argparse.ArgumentParser(description="Compare face detectors; M switches, Q exits.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--detector", choices=["haar", "mediapipe"], default="haar")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    args = parser.parse_args()
    cap = open_camera(args.camera, args.backend)
    detectors = {"Haar": HaarFaceDetector()}
    try:
        detectors["MediaPipe"] = MediaPipeFaceDetector()
    except (ImportError, AttributeError, RuntimeError) as error:
        print(f"MediaPipe unavailable: {error}. Using Haar.")
    current = "MediaPipe" if args.detector == "mediapipe" and "MediaPipe" in detectors else "Haar"
    while True:
        frame = read_camera_frame(cap)
        frame = cv2.flip(frame, 1)  # айна сияқты көрсету
        for (x, y, w, h) in detectors[current].detect(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, current, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Face detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("m"):
            current = "Haar" if current == "MediaPipe" or "MediaPipe" not in detectors else "MediaPipe"
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
