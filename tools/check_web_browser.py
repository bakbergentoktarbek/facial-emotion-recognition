"""Check the running UI through an Edge/Chrome debug port using a virtual camera.

Start a headless browser with --remote-debugging-port=9223,
--use-fake-device-for-media-stream and --use-fake-ui-for-media-stream first.
Optional --photo uses a local face image; screenshots stay in ignored reports/.
"""
import argparse
import base64
import json
from pathlib import Path
import time

import httpx
from PIL import Image
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9223)
    parser.add_argument("--photo", type=Path)
    args = parser.parse_args()
    out = ROOT / "reports" / "web_design_checks"
    out.mkdir(exist_ok=True)
    blank = out / "blank.png"
    Image.new("RGB", (320, 240), (30, 40, 55)).save(blank)
    photo = args.photo.resolve() if args.photo else blank
    pages = httpx.get(f"http://127.0.0.1:{args.port}/json/list").json()
    page = next(p for p in pages if p["type"] == "page" and "127.0.0.1:7860" in p["url"])
    seq = 0
    with connect(page["webSocketDebuggerUrl"], max_size=20_000_000) as ws:
        def call(method, params=None):
            nonlocal seq
            seq += 1
            ws.send(json.dumps({"id": seq, "method": method, "params": params or {}}))
            while True:
                result = json.loads(ws.recv())
                if result.get("id") == seq:
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    return result.get("result", {})

        def js(source):
            result = call("Runtime.evaluate", {"expression": source, "returnByValue": True})
            if "exceptionDetails" in result:
                raise RuntimeError(result["exceptionDetails"])
            return result.get("result", {}).get("value")

        def wait(source, seconds=25):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                value = js(source)
                if value:
                    return value
                time.sleep(.25)
            raise AssertionError(f"Timed out: {source}")

        def click(selector):
            box = js(f'''(() => {{const e = document.querySelector({json.dumps(selector)});
                e.scrollIntoView({{block: "center"}}); const r = e.getBoundingClientRect();
                return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};}})()''')
            call("Input.dispatchMouseEvent", {"type": "mousePressed", **box,
                                               "button": "left", "buttons": 1, "clickCount": 1})
            call("Input.dispatchMouseEvent", {"type": "mouseReleased", **box,
                                               "button": "left", "buttons": 0, "clickCount": 1})

        def tab(name):
            click(f"#{name}-tab-button")
            time.sleep(.3)

        def screenshot(name):
            js("window.scrollTo(0,0)")
            result = call("Page.captureScreenshot", {"format": "png"})
            (out / f"{name}.png").write_bytes(base64.b64decode(result["data"]))

        def upload(path):
            doc = call("DOM.getDocument")
            node = call("DOM.querySelector", {"nodeId": doc["root"]["nodeId"],
                                              "selector": "#photo-input input[type=file]"})
            call("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": [str(path)]})
            wait('Boolean(document.querySelector("#photo-input img")?.naturalWidth)')

        call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000,
                                                      "deviceScaleFactor": 1, "mobile": False})
        call("Page.reload", {"ignoreCache": True})
        time.sleep(2)
        wait('Boolean(document.querySelector("#live-result"))')
        for name in ("camera", "photo", "game"):
            tab(name)
            assert js("document.documentElement.scrollWidth <= innerWidth")
            assert js('getComputedStyle(document.querySelector("#app-header")).backgroundColor') != "rgb(255, 220, 85)"
            screenshot("desktop-" + name)
        print("OK desktop tabs, shared header and no horizontal overflow", flush=True)

        tab("photo")
        upload(photo)
        click("#analyze-photo")
        wait('Boolean(document.querySelector("#photo-result .emotion-row")) || document.querySelector("#photo-result").innerText.includes("Бет анықталмады")', 60)
        if args.photo:
            assert js('document.querySelectorAll("#photo-result .emotion-row").length') == 7
            assert js('document.querySelectorAll("#photo-result .is-leading").length') == 1
        print("OK browser photo upload and analysis", flush=True)
        # Exercise the actual dropdown rather than altering a server value.
        click("#model-picker .icon-wrap")
        wait('Boolean(document.querySelector("#model-picker [role=option]"))')
        click('#model-picker [role=option][aria-label="CNN · KDEF"]')
        wait('document.querySelector("#photo-result").innerText.includes("Нәтиже осы жерде көрсетіледі")')
        upload(blank)
        click("#analyze-photo")
        wait('document.querySelector("#photo-result").innerText.includes("Бет анықталмады")')
        assert js('document.querySelectorAll("#photo-result .emotion-row").length') == 0
        assert js('document.querySelector("#photo-tab .processing-status").innerText.includes("CNN · KDEF")')
        print("OK model change and no-face reset without stale percentages", flush=True)

        tab("camera")
        click("#start-camera")
        wait('Boolean(document.querySelector("#browser-camera video")?.readyState >= 2)')
        wait('document.querySelector("#live-result").innerText.includes("Бет анықталмады")', 45)
        click("#stop-camera")
        wait('document.querySelector("#camera-feedback").innerText.includes("Талдау тоқтатылды")')
        wait('![...document.querySelectorAll("#browser-camera [title]")].some(x=>x.title==="stop recording")')
        print("OK camera start, server result and stop", flush=True)

        tab("game")
        click("#start-game-camera")
        wait('Boolean(document.querySelector("#game-camera video")?.readyState >= 2)')
        time.sleep(1)
        click("#start-game")
        wait('!document.querySelector("#game-target").innerText.includes("Дайынсыз ба?")', 45)
        wait('document.querySelector("#game-progress").innerText.includes("Бет әлпетін сақтаңыз")')
        wait('Boolean(document.querySelector("#game-stats .urgent"))')
        wait('document.querySelector("#game-progress").innerText.includes("Раунд аяқталды")')
        screenshot("desktop-game-round")
        click("#stop-game")
        wait('document.querySelector("#game-target").innerText.includes("Дайынсыз ба?")')
        print("OK game round, urgent timer, automatic scoring and stop", flush=True)

        call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844,
                                                      "deviceScaleFactor": 1, "mobile": True})
        for name in ("camera", "photo", "game"):
            tab(name)
            assert js("document.documentElement.scrollWidth <= innerWidth"), name
            if name != "game":
                assert js(f'document.querySelector("#{name}-tab .analysis-result").getBoundingClientRect().top > document.querySelector("#{name}-tab .analysis-media").getBoundingClientRect().top')
            else:
                assert js('document.querySelector("#game-camera").getBoundingClientRect().height <= 205')
                assert js('getComputedStyle(document.querySelector("#game-stats strong")).color') == "rgb(23, 23, 23)"
                assert js('getComputedStyle(document.querySelector("#game-camera")).transform') == "none"
            screenshot("mobile-" + name)
        print("OK mobile tabs, results below media, compact game and no overflow", flush=True)
        call("Browser.close")
    print("Browser checks passed; virtual camera, local image and 390px emulation only.")


if __name__ == "__main__":
    main()
