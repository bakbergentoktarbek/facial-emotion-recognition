"""Open a camera and verify that the selected backend actually returns frames."""
import sys
import time
import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


def read_camera_frame(cap, attempts=5):
    """Allow brief frame drops while the camera/exposure is starting up."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.1)
    raise RuntimeError("Camera returned no frames after retries. Close other camera apps and retry.")


def open_camera(index=0, backend="auto"):
    if backend not in ("auto", *BACKENDS):
        raise ValueError(f"Unknown camera backend: {backend}")
    if backend == "auto":
        # Keep OpenCV's original automatic selection; use DirectShow only if it fails.
        choices = ["any", "dshow"] if sys.platform == "win32" else ["any"]
    else:
        choices = [backend]
    failures = []
    for name in choices:
        print(f"Opening camera {index}, backend {name.upper()}...", flush=True)
        cap = None
        ready = False
        try:
            cap = cv2.VideoCapture(index, BACKENDS[name])
            if not cap.isOpened():
                failures.append(f"{name}: device did not open")
                continue
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    print(f"Camera ready: {cap.getBackendName()}, "
                          f"{frame.shape[1]}x{frame.shape[0]}", flush=True)
                    if float(frame.mean()) < 1.0:
                        print("Warning: camera frames are almost black before model processing. "
                              "Check the camera preview, privacy shutter and lighting.", flush=True)
                    ready = True
                    return cap
                time.sleep(0.1)
            failures.append(f"{name}: device opened but returned no frames")
        except cv2.error as error:
            failures.append(f"{name}: {error}")
        finally:
            if cap is not None and not ready:
                cap.release()
                time.sleep(0.5)
    raise RuntimeError(
        f"Camera {index} unavailable ({'; '.join(failures)}). "
        "Close other camera apps, check Windows camera permissions, "
        "or select another --camera index / --backend dshow."
    )
