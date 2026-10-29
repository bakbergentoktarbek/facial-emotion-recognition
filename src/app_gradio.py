# src/app_gradio.py
import cv2
import gradio as gr
from config import EMOTIONS_EN, EMOTIONS_KZ, MODEL_PATH
from face_detect import get_detector
from predictor import EmotionPredictor, crop_face, draw_label, put_text

detector = get_detector("mediapipe")
predictor = EmotionPredictor(MODEL_PATH)


def analyze(image_rgb):
    """Gradio RGB сурет береді; біздің функциялар BGR күтеді."""
    if image_rgb is None:
        return None, {}
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    boxes = detector.detect(frame)
    if not boxes:
        frame = put_text(frame, "Бет табылмады", (10, 10), 26, bg=(192, 57, 43))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {}
    main_probs = None
    for box in sorted(boxes, key=lambda b: b[2] * b[3]):  # соңғысы — ең үлкен бет
        emotion, conf, probs = predictor.predict(crop_face(frame, box))
        frame = draw_label(frame, box, emotion, conf)
        main_probs = probs
    scores = {EMOTIONS_KZ[e]: float(p) for e, p in zip(EMOTIONS_EN, main_probs)}
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), scores


def analyze_stream(image_rgb):
    return analyze(image_rgb)[0]


with gr.Blocks(title="Эмоцияны тану") as demo:
    gr.Markdown("# Бет-бейнеден эмоцияны тану\n"
                "Оқу жобасы. Модель бет өрнегін жіктейді. "
                "нәтижені адамның шын сезімі туралы қорытынды ретінде қабылдамаңыз. "
                "Суреттер сақталмайды.")
    with gr.Tab("Сурет жүктеу"):
        with gr.Row():
            inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Кіріс")
            out_img = gr.Image(label="Нәтиже")
        out_lbl = gr.Label(num_top_classes=3, label="Ең үлкен беттің эмоциялары")
        gr.Button("Талдау", variant="primary").click(analyze, inp, [out_img, out_lbl])
    with gr.Tab("Тікелей веб-камера"):
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Камера")
            live = gr.Image(label="Нәтиже")
        cam.stream(analyze_stream, inputs=cam, outputs=live, stream_every=0.2)

if __name__ == "__main__":
    demo.launch(share=False)  # топқа көрсету үшін share=True
