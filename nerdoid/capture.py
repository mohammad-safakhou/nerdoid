from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class FrameMeta:
    width: int
    height: int
    scale: float


CapturedFrame = Tuple[np.ndarray, FrameMeta]


class ScreenCapturer:
    def __init__(self, fps: int, downscale_width: int):
        self._fps = fps
        self._downscale_width = downscale_width
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, frame_queue: queue.Queue[CapturedFrame]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(frame_queue,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self, frame_queue: queue.Queue[CapturedFrame]) -> None:
        period = 1.0 / float(self._fps)
        with mss.mss() as sct:
            mon = sct.monitors[1]
            mon_w, mon_h = mon["width"], mon["height"]
            while not self._stop.is_set():
                t0 = time.time()
                raw = sct.grab(mon)
                img = np.array(raw)[:, :, :3]
                h, w = img.shape[:2]

                scale = 1.0
                if w > self._downscale_width:
                    scale = self._downscale_width / float(w)
                    img = cv2.resize(
                        img,
                        (self._downscale_width, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                meta = FrameMeta(width=mon_w, height=mon_h, scale=scale)
                try:
                    frame_queue.put_nowait((img, meta))
                except queue.Full:
                    pass

                dt = time.time() - t0
                time.sleep(max(0.0, period - dt))
