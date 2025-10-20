from __future__ import annotations

import time
from typing import Dict

import pyautogui


class Actuator:
    def __init__(self, click_min_period: float):
        pyautogui.FAILSAFE = False
        self._click_min_period = click_min_period
        self._last_click = 0.0

    def execute(self, cmd: Dict) -> None:
        action = cmd.get("action", "none")
        if action == "move_mouse":
            self._move_mouse(cmd)
        elif action == "click":
            self._click(cmd)

    def _move_mouse(self, cmd: Dict) -> None:
        try:
            pyautogui.moveTo(cmd["x"], cmd["y"], duration=0)
        except Exception as exc:  # pragma: no cover - pyautogui system dependency
            print("[actuator] move_mouse error:", exc)

    def _click(self, cmd: Dict) -> None:
        if time.time() - self._last_click < self._click_min_period:
            return
        try:
            pyautogui.click(button=cmd.get("button", "left"))
            self._last_click = time.time()
        except Exception as exc:  # pragma: no cover - pyautogui system dependency
            print("[actuator] click error:", exc)
