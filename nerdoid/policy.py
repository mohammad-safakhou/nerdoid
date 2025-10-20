from __future__ import annotations

import json
from typing import Dict, List, Optional

import cv2
import numpy as np
import pyautogui
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .capture import FrameMeta
from .settings import RuntimeConfig

SYSTEM_INSTRUCTION = (
    "You are a UI perception agent. "
    "Given a single screen image, identify the main interactive target under or near the mouse "
    "and propose ONE atomic action as strict JSON ONLY with this schema:\n"
    '{"action":"move_mouse","x":123,"y":456}\n'
    "or\n"
    '{"action":"click","button":"left"}\n'
    "or\n"
    '{"action":"none"}\n'
    "Rules: Respond with ONLY one JSON object. No prose. Coordinates are absolute screen pixels."
)


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _annotate_mouse(img: np.ndarray, mx: int, my: int, scale: float) -> tuple[np.ndarray, tuple[int, int]]:
    if mx < 0 or my < 0:
        return img, (-1, -1)
    annotated = img.copy()
    dx, dy = int(mx * scale), int(my * scale)
    h, w = annotated.shape[:2]
    if 0 <= dx < w and 0 <= dy < h:
        cv2.drawMarker(
            annotated,
            (dx, dy),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        return annotated, (dx, dy)
    return annotated, (-1, -1)


def _extract_first_json(text: str) -> Optional[Dict]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        if text[idx] != "{":
            idx += 1
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue
        return obj
    return None


class VisionAgent:
    def __init__(self, config: RuntimeConfig):
        self._config = config
        torch.set_grad_enabled(False)
        self._processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            dtype=config.dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(config.device)
        self._model.eval()
        self._recent_outputs: List[str] = []

    def decide(self, frame_bgr: np.ndarray, meta: FrameMeta) -> Dict:
        try:
            mx, my = pyautogui.position()
        except Exception:
            mx, my = -1, -1

        annotated, down_mouse = _annotate_mouse(frame_bgr, mx, my, meta.scale)
        image = _to_pil(annotated)

        system = (
            SYSTEM_INSTRUCTION
            + f"\nScreen width={meta.width}, height={meta.height}, downscale={meta.scale:.4f}."
            + f"\nCanvas width={image.width}, height={image.height}. Mouse (downscaled)={down_mouse}."
        )
        if self._config.task_prompt:
            system += f"\nCurrent task: {self._config.task_prompt}"
        if self._recent_outputs:
            history = "\n".join(f"- {entry}" for entry in self._recent_outputs[-self._config.max_action_history :])
            system += f"\nRecent agent outputs:\n{history}"

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Decide one atomic action for THIS frame only."},
                ],
            },
        ]

        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

        out = self._model.generate(
            **inputs,
            max_new_tokens=self._config.max_new_tokens,
            do_sample=False,
            eos_token_id=self._processor.tokenizer.eos_token_id
            if hasattr(self._processor, "tokenizer")
            else None,
        )

        text = self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        obj = _extract_first_json(text)
        if obj is None:
            self._record_output(text or "<no JSON>", {"action": "none"})
            return {"action": "none"}

        action = obj.get("action")
        if action == "move_mouse" and "x" in obj and "y" in obj:
            cmd = {"action": "move_mouse", "x": int(obj["x"]), "y": int(obj["y"])}
            self._record_output(text, cmd)
            return cmd
        if action == "click":
            btn = obj.get("button", "left")
            if btn not in ["left", "right", "middle"]:
                btn = "left"
            cmd = {"action": "click", "button": btn}
            self._record_output(text, cmd)
            return cmd

        self._record_output(text, {"action": "none"})
        return {"action": "none"}

    def _record_output(self, raw_text: str, cmd: Dict) -> None:
        summary = raw_text.strip()
        if not summary:
            summary = json.dumps(cmd)
        if len(summary) > 240:
            summary = summary[:237] + "..."
        self._recent_outputs.append(summary)
        if len(self._recent_outputs) > self._config.max_action_history:
            self._recent_outputs = self._recent_outputs[-self._config.max_action_history :]
