"""Local Gradio interface. Run: python src/app_gradio.py."""
import os
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
import threading
import time

from config import BASE_DIR, EMOTIONS_EN, EMOTIONS_KZ, MODEL_ALIASES

# Gradio uses temporary files for browser transport, but never records video.
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(BASE_DIR) / "reports" / "gradio_cache"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
from gradio.route_utils import delete_files_created_by_app
import numpy as np
from face_detect import get_detector
from predictor import EmotionPredictor, crop_face, draw_label, put_text
from web_game import WebGameSession
from ui_components import game_panels, result_panel

DEFAULT_MODEL = "mobilenet-kdef"
MODEL_NAMES = {
    "mobilenet-kdef": "MobileNetV2 · KDEF",
    "cnn-kdef": "CNN · KDEF",
    "cnn-fer": "CNN · FER-2013",
    "mobilenet-fer": "MobileNetV2 · FER-2013",
}
MODEL_RESULTS = {
    "mobilenet-kdef": "83,33% · KDEF тесті, 768 сурет · RGB 160×160",
    "cnn-kdef": "80,08% · KDEF тесті, 768 сурет · сұр кескін 96×96",
    "cnn-fer": "61,56% · FER-2013 тесті, 7 178 сурет · сұр кескін 48×48",
    "mobilenet-fer": "47,73% · FER-2013 тесті, 7 178 сурет · сұр кескін 48×48",
}


def model_details(alias):
    return (f"**{MODEL_NAMES[alias]}** — {MODEL_RESULTS[alias]}.\n\n"
            "Бұл тест жиынындағы дәлдік. Камерадағы нәтиже жарық пен ракурсқа тәуелді. "
            "Әртүрлі датасеттегі пайыздарды тікелей салыстыруға болмайды.")


class WebRecognizer:
    """Reuse warmed models; serialize access to the shared MediaPipe detector."""

    def __init__(self):
        self.detector = None
        self.predictors = {}
        self.lock = threading.Lock()

    def warm_up(self, alias):
        if alias not in MODEL_ALIASES:
            raise ValueError("Белгісіз модель. Тізімнен модель таңдаңыз.")
        with self.lock:
            if alias not in self.predictors:
                self.predictors[alias] = EmotionPredictor(alias)

    def analyze(self, image_rgb, alias=DEFAULT_MODEL, max_side=960):
        if alias not in MODEL_ALIASES:
            raise ValueError("Белгісіз модель. Тізімнен модель таңдаңыз.")
        if image_rgb is None:
            return None, {}, "Сурет жүктеңіз немесе камераны қосыңыз."
        if (not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8 or
                image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.size == 0):
            raise ValueError("RGB суретін жүктеңіз.")

        with self.lock:
            started = time.perf_counter()
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            if self.detector is None:
                self.detector = get_detector("mediapipe")
            boxes = self.detector.detect(frame)
            predictions = []
            if boxes:
                if alias not in self.predictors:
                    self.predictors[alias] = EmotionPredictor(alias)
                predictor = self.predictors[alias]
                # Complete all predictions before drawing overlapping labels.
                for box in sorted(boxes, key=lambda b: b[2] * b[3]):
                    face = crop_face(frame, box)
                    if face.size:
                        emotion, confidence, probs = predictor.predict(face)
                        predictions.append((box, emotion, confidence, probs))
            scores = {}
            if predictions:
                for box, emotion, confidence, _ in predictions:
                    frame = draw_label(frame, box, emotion, confidence)
                scores = {EMOTIONS_KZ[e]: float(p)
                          for e, p in zip(EMOTIONS_EN, predictions[-1][3])}
                message = f"Табылған бет: {len(predictions)}. Ықтималдықтар ең үлкен бет үшін көрсетілген."
            else:
                frame = put_text(frame, "Бет табылмады", (10, 10), 26, bg=(192, 57, 43))
                message = "Бет табылмады. Камераға тура қарап, жарықты тексеріңіз."
            elapsed = (time.perf_counter() - started) * 1000
            status = f"{MODEL_NAMES[alias]} · {elapsed:.0f} мс\n\n{message}"
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), scores, status


recognizer = WebRecognizer()


def _analyze(image_rgb, alias, max_side):
    try:
        return recognizer.analyze(image_rgb, alias, max_side)
    except FileNotFoundError as error:
        raise gr.Error("Модель файлы табылмады. README нұсқаулығымен GitHub Releases-тен "
                       "жүктеп, models/ бумасына салыңыз.") from error
    except ValueError as error:
        raise gr.Error(str(error)) from error


def analyze(image_rgb, alias=DEFAULT_MODEL):
    return _analyze(image_rgb, alias, 960)


def analyze_stream(image_rgb, alias=DEFAULT_MODEL):
    return _analyze(image_rgb, alias, 640)


def start_web_game(alias, session):
    # Streaming Image events carry frames separately from the component's
    # value; a button click may see value=None even with a working camera.
    if session.camera_frame_at is None or time.perf_counter() - session.camera_frame_at > 3:
        raise gr.Error("Алдымен ойын камерасын «Ойын камерасын қосу» батырмасымен қосыңыз.")
    try:
        # Loading time must not consume the player's first round.
        recognizer.warm_up(alias)
    except FileNotFoundError as error:
        raise gr.Error("Модель файлын GitHub Releases-тен жүктеп, models/ бумасына салыңыз.") from error
    session.begin(alias)
    return session.summary(MODEL_NAMES[alias]), session


def stop_web_game(session):
    session.stop()
    return session.summary(), session


def analyze_game_frame(image_rgb, session):
    if image_rgb is None:
        return None, session.summary(), session
    height, width = image_rgb.shape[:2]
    if max(height, width) > 640:
        scale = 640 / max(height, width)
        image_rgb = cv2.resize(image_rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
    session.camera_frame_at = time.perf_counter()
    if not session.active:
        return image_rgb, session.summary(MODEL_NAMES.get(session.model_alias, "")), session
    run_id = session.run_id
    annotated, scores, _ = _analyze(image_rgb, session.model_alias, 640)
    if session.run_id != run_id:
        return image_rgb, session.summary(MODEL_NAMES.get(session.model_alias, "")), session
    session.consume(scores)
    summary = session.summary(MODEL_NAMES[session.model_alias])
    if not scores and session.state == "play":
        summary += "\n\nБет табылмады. Камераға тура қараңыз."
    return annotated, summary, session


def camera_start_js(element_id):
    return Path(__file__).with_name("web_camera.js").read_text(encoding="utf-8").replace(
        '"browser-camera"', f'"{element_id}"')


@asynccontextmanager
async def cache_lifespan(app):
    async def cleanup():
        while True:
            await asyncio.sleep(60)
            try:
                await asyncio.to_thread(delete_files_created_by_app, app.get_blocks(), 60)
            except OSError:
                # Windows may still be serving an image. Retry next time;
                # a locked file must not terminate all future cleanup.
                print("Image cache is busy; cleanup will retry in 60 seconds.", flush=True)
    task = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with suppress(OSError):
            await asyncio.to_thread(delete_files_created_by_app, app.get_blocks(), None)


def analyze_photo_ui(image_rgb, alias=DEFAULT_MODEL):
    frame, scores, status = analyze(image_rgb, alias)
    return frame, scores, status, result_panel(scores, image_rgb is not None, MODEL_NAMES[alias])


def analyze_camera_ui(image_rgb, alias=DEFAULT_MODEL):
    frame, scores, status = analyze_stream(image_rgb, alias)
    return frame, scores, status, result_panel(scores, image_rgb is not None, MODEL_NAMES[alias])


def start_game_ui(alias, session):
    board, session = start_web_game(alias, session)
    return board, session, *game_panels(session, MODEL_NAMES[session.model_alias])


def stop_game_ui(session):
    board, session = stop_web_game(session)
    return board, session, *game_panels(session)


def game_frame_ui(image_rgb, session):
    frame, board, session = analyze_game_frame(image_rgb, session)
    return frame, board, session, *game_panels(session, MODEL_NAMES.get(session.model_alias, ""))


def camera_stop_js(element_id):
    return """() => {
        const root = document.getElementById("%s");
        root?.querySelector('[title="stop recording"]')?.closest("button")?.click();
        return "Талдау тоқтатылды. Жалғастыру үшін камераны қосу батырмасын басыңыз.";
    }""" % element_id


def build_demo():
    theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate",
                           font=["Segoe UI", "Arial", "sans-serif"]).set(
        body_background_fill="#0b1423", body_background_fill_dark="#0b1423",
        body_text_color="#edf3fa", body_text_color_dark="#edf3fa",
        block_background_fill="#121f32", block_background_fill_dark="#121f32",
        background_fill_primary="#121f32", background_fill_primary_dark="#121f32",
        background_fill_secondary="#18283e", background_fill_secondary_dark="#18283e",
        input_background_fill="#15243a", input_background_fill_dark="#15243a",
        block_border_color="#293b53", block_border_color_dark="#293b53",
        border_color_primary="#293b53", border_color_primary_dark="#293b53",
        block_label_background_fill="#17283e", block_label_background_fill_dark="#17283e",
        block_label_text_color="#b9cce0", block_label_text_color_dark="#b9cce0",
        button_primary_background_fill="#51ddbd", button_primary_background_fill_dark="#51ddbd",
        button_primary_text_color="#08241f", button_primary_text_color_dark="#08241f",
        button_secondary_background_fill="#1b2c43", button_secondary_background_fill_dark="#1b2c43",
        button_secondary_text_color="#dfebf8", button_secondary_text_color_dark="#dfebf8",
        input_radius="10px", block_radius="14px",
    )
    with gr.Blocks(title="Mimika Lab · Бет әлпетін талдау", theme=theme,
                   css=Path(__file__).with_name("web_styles.css").read_text(encoding="utf-8"),
                   js=Path(__file__).with_name("web_ui.js").read_text(encoding="utf-8"),
                   analytics_enabled=False) as demo:
        with gr.Row(elem_id="app-header"):
            with gr.Column(elem_id="brand-column"):
                gr.HTML('<div class="brand"><div class="brand-mark" aria-hidden="true">m.</div>'
                        '<div><div class="brand-name">Mimika Lab</div>'
                        '<div class="brand-subtitle">Бет әлпетін талдау</div></div></div>')
            with gr.Column(elem_id="model-column"):
                model = gr.Dropdown(choices=[(name, alias) for alias, name in MODEL_NAMES.items()],
                                    value=DEFAULT_MODEL, label="Модель", interactive=True,
                                    elem_id="model-picker")

        with gr.Tabs(elem_id="workspace"):
            with gr.Tab("Камера", elem_id="camera-tab"):
                gr.HTML('<div class="section-title"><div class="eyebrow">ОСЫ СӘТТЕ · ОСЫ ЖЕРДЕ</div>'
                        '<h1>Бет әлпетіңіз не дейді?</h1>'
                        '<p>Камераны қосып, жеті эмоция санатының модель бағасын көріңіз.</p></div>')
                with gr.Row(elem_classes="analysis-layout"):
                    with gr.Column(scale=7, elem_classes="analysis-media"):
                        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                                       label="Камера", height=360, elem_id="browser-camera",
                                       elem_classes="camera-card",
                                       webcam_constraints={"width": {"ideal": 640}, "height": {"ideal": 480}})
                        with gr.Row(elem_classes="action-row"):
                            start_camera = gr.Button("Камераны қосу", variant="primary", elem_id="start-camera")
                            stop_camera = gr.Button("Талдауды тоқтату", elem_id="stop-camera")
                        camera_feedback = gr.Markdown("Камераға рұқсат беріңіз. Бетіңіз толық көрінсін.",
                                                      elem_id="camera-feedback", elem_classes="camera-feedback")
                        with gr.Accordion("Белгіленген кадр", open=False, elem_classes="processed-frame"):
                            live = gr.Image(label="Бет жақтауы бар кадр", height=320, interactive=False,
                                            show_download_button=False, elem_id="live-annotated")
                    with gr.Column(scale=4, elem_classes="analysis-result"):
                        live_panel = gr.HTML(result_panel(), elem_id="live-result")
                        live_status = gr.Markdown("", elem_classes="processing-status")
                        live_lbl = gr.Label(visible=False, num_top_classes=7)
                start_camera.click(fn=None, outputs=camera_feedback, queue=False,
                                   js=camera_start_js("browser-camera"))
                stop_camera.click(fn=None, outputs=camera_feedback, queue=False,
                                  js=camera_stop_js("browser-camera"))
                cam.stream(analyze_camera_ui, [cam, model], [live, live_lbl, live_status, live_panel],
                           stream_every=0.15, time_limit=120, concurrency_id="live-camera",
                           concurrency_limit=1, trigger_mode="always_last",
                           show_progress="hidden", api_name="analyze_camera")

            with gr.Tab("Сурет", elem_id="photo-tab"):
                gr.HTML('<div class="section-title"><div class="eyebrow">БІР СУРЕТ · ЖЕТІ САНАТ</div>'
                        '<h1>Бір сәтке жақынырақ қараңыз.</h1>'
                        '<p>Сурет жүктеңіз. Модель әр бетті талдап, ең үлкен бет үшін бағаларды көрсетеді.</p></div>')
                with gr.Row(elem_classes="analysis-layout"):
                    with gr.Column(scale=7, elem_classes="analysis-media"):
                        inp = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Сурет жүктеу",
                                       height=360, elem_id="photo-input", elem_classes="photo-card")
                        button = gr.Button("Суретті талдау", variant="primary", elem_id="analyze-photo")
                        with gr.Accordion("Белгіленген кадр", open=False, elem_classes="processed-frame"):
                            out_img = gr.Image(label="Бет жақтауы бар сурет", height=320, interactive=False,
                                               elem_id="photo-annotated")
                    with gr.Column(scale=4, elem_classes="analysis-result"):
                        photo_panel = gr.HTML(result_panel(), elem_id="photo-result")
                        photo_status = gr.Markdown("", elem_classes="processing-status")
                        out_lbl = gr.Label(visible=False, num_top_classes=7)
                button.click(analyze_photo_ui, [inp, model], [out_img, out_lbl, photo_status, photo_panel],
                             api_name="analyze_photo", concurrency_id="photos", concurrency_limit=1)
                inp.clear(lambda: (None, {}, "", result_panel()),
                          outputs=[out_img, out_lbl, photo_status, photo_panel], queue=False, api_name=False)

            with gr.Tab("Ойын", elem_id="game-tab"):
                gr.HTML('<div class="game-heading"><h1 class="game-title">Эмоцияны қайтала!</h1>'
                        '<span class="pink-sticker">БЕТІҢІЗБЕН ОЙНАҢЫЗ ✦</span></div>')
                game_session = gr.State(WebGameSession())
                stats, target, progress = game_panels(WebGameSession())
                game_stats = gr.HTML(stats, elem_id="game-stats")
                with gr.Row(elem_classes="game-layout"):
                    with gr.Column(scale=4, elem_classes="game-task-column"):
                        game_target = gr.HTML(target, elem_id="game-target")
                    with gr.Column(scale=6, elem_classes="game-camera-column"):
                        game_cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                                            label="Ойын камерасы", height=290, elem_id="game-camera",
                                            elem_classes="camera-card",
                                            webcam_constraints={"width": {"ideal": 640}, "height": {"ideal": 480}})
                        game_progress = gr.HTML(progress, elem_id="game-progress")
                        game_camera_button = gr.Button("Ойын камерасын қосу", elem_id="start-game-camera")
                        with gr.Row(elem_classes="action-row"):
                            game_start = gr.Button("Ойынды бастау / қайта ойнау", variant="primary", elem_id="start-game")
                            game_stop = gr.Button("Ойынды тоқтату", elem_id="stop-game")
                        game_feedback = gr.Markdown("Алдымен камераны қосып, содан кейін ойынды бастаңыз.",
                                                   elem_id="game-camera-feedback", elem_classes="camera-feedback")
                        with gr.Accordion("Белгіленген кадр", open=False, elem_classes="processed-frame"):
                            game_view = gr.Image(label="Бет жақтауы бар ойын кадры", height=280,
                                                 interactive=False, show_download_button=False)
                game_board = gr.Markdown(WebGameSession().summary(), elem_id="game-board", visible=False)
                game_camera_button.click(fn=None, outputs=game_feedback, queue=False,
                                         js=camera_start_js("game-camera"))
                game_start.click(start_game_ui, [model, game_session],
                                 [game_board, game_session, game_stats, game_target, game_progress],
                                 queue=False, api_name="start_game")
                game_stop.click(stop_game_ui, game_session,
                                [game_board, game_session, game_stats, game_target, game_progress],
                                queue=False, api_name="stop_game")
                game_cam.stream(game_frame_ui, [game_cam, game_session],
                                [game_view, game_board, game_session, game_stats, game_target, game_progress],
                                stream_every=0.15, time_limit=120, concurrency_id="game-camera",
                                concurrency_limit=1, trigger_mode="always_last", show_progress="hidden",
                                api_name="game_frame")
                gr.Markdown("**5 раунд · 2 с дайындық · 5 с көрсету · максимум 600 ұпай.** "
                            "Раунд автоматты аяқталады. Ойын басталғанда таңдалған модель сақталады. "
                            "Рекорд осы сессияға тиесілі. Ұпай модельдің дәлдігі емес.",
                            elem_classes="game-rules")
        with gr.Accordion("Модель туралы", open=False, elem_id="model-about"):
            details = gr.Markdown(model_details(DEFAULT_MODEL))
            gr.Markdown("Модель бет өрнегін жеті санатқа жіктейді. Бағалар — адамның нақты сезімі туралы "
                        "қорытынды емес. Алғашқы талдауда модель жүктеледі, кейін қайта қолданылады. "
                        "Өңдеу уақыты браузермен дерек алмасуды қамтымайды. Кадрлар келмесе, ойын таймері "
                        "уақытша тоқтайды. Gradio уақытша сурет файлдарын жергілікті кэште сақтайды; "
                        "ескірген файлдар мерзімді тазаланады.")
        def change_model(alias):
            return model_details(alias), result_panel(), result_panel(), {}, {}, "", ""
        model.change(change_model, model, [details, photo_panel, live_panel, out_lbl, live_lbl,
                                          photo_status, live_status], queue=False, api_name=False)
        gr.HTML('<p id="privacy-note">Оқу жобасы · Камераны қатысушының келісімімен қосыңыз. '
                'Бет әлпеті адамның шынайы сезімін анықтамайды.</p>')
    return demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860, share=False,
                        max_file_size="10mb", show_error=False,
                        app_kwargs={"lifespan": cache_lifespan})
