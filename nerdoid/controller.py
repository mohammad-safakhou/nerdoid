from __future__ import annotations

import queue
import time
from typing import Dict

from .actuator import Actuator
from .capture import FrameMeta, ScreenCapturer
from .policy import VisionAgent
from .settings import RuntimeConfig


class Controller:
    def __init__(
        self,
        capture: ScreenCapturer,
        policy: VisionAgent,
        actuator: Actuator,
        config: RuntimeConfig,
    ):
        self._capture = capture
        self._policy = policy
        self._actuator = actuator
        self._config = config
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)

    def run(self) -> None:
        self._capture.start(self._frame_queue)
        latest_cmd: Dict = {"action": "none"}
        latest_meta = FrameMeta(width=1, height=1, scale=1.0)
        last_decision = 0.0

        try:
            while True:
                now = time.time()
                if (now - last_decision) >= self._config.decision_period:
                    try:
                        frame, meta = self._frame_queue.get(timeout=0.01)
                    except queue.Empty:
                        pass
                    else:
                        latest_meta = meta
                        latest_cmd = self._policy.decide(frame, meta)
                        last_decision = now
                        print("[decide]", latest_cmd, "meta=", latest_meta)

                action = latest_cmd.get("action", "none")
                if action == "move_mouse":
                    x = int(latest_cmd["x"] / latest_meta.scale)
                    y = int(latest_cmd["y"] / latest_meta.scale)
                    self._actuator.execute({"action": "move_mouse", "x": x, "y": y})
                else:
                    self._actuator.execute(latest_cmd)

                time.sleep(0.01)
        finally:
            self._capture.stop()
