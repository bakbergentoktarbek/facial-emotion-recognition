"""Presentation only: render the recognizer's probabilities and existing game state."""
from html import escape
import time

from config import EMOTIONS_EN, EMOTIONS_KZ
from game import PREP_TIME, PLAY_TIME, RESULT_TIME, ROUNDS


HINTS = {
    "happy": "Ауыз бұрыштарын көтеріп, жымиыңыз.",
    "sad": "Қастың ішкі жағын көтеріп, ауыз бұрыштарын төмен түсіріңіз.",
    "angry": "Қасыңызды жақындатып, ерніңізді қысыңыз.",
    "surprise": "Көзіңіз бен аузыңызды сәл кең ашыңыз.",
    "fear": "Көзіңізді кең ашып, қасыңызды көтеріңіз.",
    "neutral": "Бетіңізді бос ұстап, камераға тура қараңыз.",
}


def face_sticker(emotion="neutral"):
    mouths = {
        "happy": '<path d="M58 95 Q85 130 112 95 Z" fill="#161616"/>',
        "sad": '<path d="M59 114 Q85 91 111 114"/>',
        "angry": '<path d="M64 111 H106 M55 59 L72 65 M98 65 L115 59"/>',
        "surprise": '<ellipse cx="85" cy="108" rx="14" ry="20" fill="#161616"/>',
        "fear": '<ellipse cx="85" cy="109" rx="21" ry="12" fill="#161616"/><path d="M52 57 L70 51 M100 51 L118 57"/>',
        "neutral": '<path d="M65 108 H105"/>',
    }
    return ('<svg viewBox="0 0 170 170" aria-hidden="true" class="face-sticker">'
            '<path d="M85 7 L101 17 L121 14 L131 32 L151 39 L151 61 L164 78 L155 97 '
            'L157 119 L137 130 L128 151 L107 151 L85 164 L66 154 L44 155 L33 136 '
            'L14 127 L15 104 L5 85 L16 65 L15 43 L34 34 L44 14 L65 17 Z" '
            'fill="#b9c6ff" stroke="#161616" stroke-width="3"/>'
            '<circle cx="85" cy="85" r="56" fill="#fff9df" stroke="#161616" stroke-width="3"/>'
            '<g stroke="#161616" stroke-width="4" stroke-linecap="round" fill="none">'
            '<path d="M62 75 V81 M108 75 V81"/>' + mouths.get(emotion, mouths["neutral"]) + '</g></svg>')


def result_panel(scores=None, analyzed=False, model_name=""):
    title = '<div class="panel-eyebrow">ТАЛДАУ НӘТИЖЕСІ</div>'
    if not scores:
        message = "Бет анықталмады" if analyzed else "Нәтиже осы жерде көрсетіледі"
        detail = "Камераға тура қарап, жарықты тексеріңіз." if analyzed else "Камераны қосыңыз немесе сурет жүктеңіз."
        return (f'<section class="result-card result-empty">{title}<div class="empty-face">'
                f'{face_sticker()}</div><h2>{message}</h2><p>{detail}</p></section>')
    values = {e: max(0.0, min(1.0, float(scores[EMOTIONS_KZ[e]]))) for e in EMOTIONS_EN}
    leader = max(values, key=values.get)
    rows = []
    for emotion, value in values.items():
        percent = value * 100
        selected = " is-leading" if emotion == leader else ""
        rows.append(f'<div class="emotion-row{selected}"><div class="emotion-caption">'
                    f'<span>{EMOTIONS_KZ[emotion]}</span><span>{percent:.1f}%</span></div>'
                    f'<div class="emotion-track"><i style="width:{percent:.3f}%"></i></div></div>')
    return (f'<section class="result-card">{title}<div class="leading-result">'
            f'<h2>{EMOTIONS_KZ[leader]}</h2><strong>{values[leader] * 100:.0f}<small>%</small></strong></div>'
            '<p class="result-caption">Модельдің ең жоғары бағасы</p><div class="emotion-bars">'
            + ''.join(rows) + f'</div><div class="result-source">{escape(model_name)}'
            ' · Ең үлкен бет бойынша</div></section>')


def game_panels(session, model_name="", now=None):
    now = time.perf_counter() if now is None else now
    duration = {"prep": PREP_TIME, "play": PLAY_TIME, "result": RESULT_TIME}.get(session.state)
    left = max(0.0, duration - (now - session.t0)) if duration is not None else None
    timer = f"{left:.1f}<small> с</small>" if left is not None else "—"
    urgent = " urgent" if session.state == "play" and left <= 2 else ""
    stats = (f'<div class="game-stats"><div><span>РАУНД</span><strong>{session.round}<small> / {ROUNDS}</small></strong></div>'
             f'<div><span>ҰПАЙ</span><strong>{session.total}<small> / {ROUNDS * 120}</small></strong></div>'
             f'<div class="game-timer{urgent}"><span>УАҚЫТ</span><strong>{timer}</strong></div>'
             f'<div class="record-stat"><span>РЕКОРД</span><strong>{session.best_score}</strong></div></div>')
    if session.state == "menu":
        emotion, title = "neutral", "Дайынсыз ба?"
        hint = "Камераны қосып, ойынды бастаңыз. Әр раундта берілген бет әлпетін қайталаңыз."
        tag = "ӨЗІҢІЗДІ СЫНАП КӨРІҢІЗ"
    elif session.state == "over":
        emotion, title = "happy", "Жарайсыз!"
        hint = "Бес раунд аяқталды. Раунд ұпайлары: " + ", ".join(map(str, session.scores)) + "."
        tag = "ОЙЫН АЯҚТАЛДЫ!"
    else:
        emotion, title = session.target, EMOTIONS_KZ[session.target]
        hint, tag = HINTS[emotion], "ОСЫ БЕТ ӘЛПЕТІН ҚАЙТАЛАҢЫЗ"
    target = (f'<article class="target-card"><span class="task-tag">{tag}</span>{face_sticker(emotion)}'
              f'<h2>{title}</h2><p>{hint}</p><span class="bonus-sticker">1 СЕКУНД = +20 БОНУС</span></article>')
    probability = session.current_probability if session.state in ("play", "result") else 0.0
    percent = max(0.0, min(1.0, probability)) * 100
    if session.state == "play":
        status = "Бет әлпетін сақтаңыз" if not session.bonus_earned else "Бонус +20 алынды!"
    elif session.state == "prep":
        status = "Дайындалыңыз…"
    elif session.state == "result":
        status = f"Раунд аяқталды: +{session.scores[-1]} ұпай"
    elif session.state == "over":
        status = f"Ойын аяқталды! Нәтиже: {session.total} ұпай"
    else:
        status = "Камераны қосып, ойынды бастаңыз"
    progress = (f'<div class="game-progress"><div><span>Модель бағасы</span><strong>{percent:.0f}%</strong></div>'
                f'<div class="game-track" role="progressbar" aria-label="Модель бағасы" aria-valuemin="0" '
                f'aria-valuemax="100" aria-valuenow="{percent:.1f}"><i style="width:{percent:.3f}%"></i></div>'
                f'<p class="hold-status">{status}</p><small>{escape(model_name)}</small></div>')
    return stats, target, progress
