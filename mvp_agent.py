import os
import json, time, threading, queue
from typing import Optional, Tuple
import numpy as np
import mss
import cv2
from PIL import Image
import pyautogui

import torch

# Robust import for AutoProcessor across transformers versions
from transformers import AutoProcessor
from transformers import AutoModelForVision2Seq

# -------- Config --------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # official, small + fast on M-series
FRAME_FPS = 10  # screen capture rate
DECISION_HZ = 4  # how often the model decides (per second)
MAX_NEW_TOKENS = 96  # keep outputs short
DOWNSCALE_WIDTH = 1280  # resize frames for speed

# macOS tip: prevent pyautogui fails on retina scaling
pyautogui.FAILSAFE = False

prefer_mps = os.environ.get("NERDOID_USE_MPS", "0") == "1"
device = "mps" if prefer_mps and torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

print(f"[boot] loading {MODEL_ID} on {device} ({dtype}) ...")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=dtype,
    low_cpu_mem_usage=True
)
model.to(device)
model.eval()

# Simple, strict JSON tool schema
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

# ------------ Screen capture ------------
frame_q = queue.Queue(maxsize=2)


# ------------ VLM decide ------------
def to_pil_bgr(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def extract_first_json(text: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue
        return obj
    return None


# ------------ Actuator ------------
last_click_t = 0.0
CLICK_MIN_PERIOD = 0.3  # seconds


def act(cmd: dict):
    global last_click_t
    a = cmd.get("action", "none")
    if a == "move_mouse":
        try:
            pyautogui.moveTo(cmd["x"], cmd["y"], duration=0)
        except Exception as e:
            print("[act] move_mouse error:", e)
    elif a == "click":
        if time.time() - last_click_t < CLICK_MIN_PERIOD:
            return
        try:
            pyautogui.click(button=cmd.get("button", "left"))
            last_click_t = time.time()
        except Exception as e:
            print("[act] click error:", e)


# ------------ Controller loop ------------
def controller():
    last_decision = 0.0
    latest_cmd = {"action": "none"}
    latest_meta = (1, 1, 1.0)
    decision_period = 1.0 / DECISION_HZ

    while True:
        now = time.time()
        if (now - last_decision) >= decision_period and not frame_q.empty():
            frame, meta = frame_q.get()
            latest_meta = meta
            latest_cmd = decide_action(frame, meta)
            last_decision = now
            print("[decide]", latest_cmd, "meta=", latest_meta)

        # rescale if needed before acting
        a = latest_cmd.get("action", "none")
        if a == "move_mouse":
            mon_w, mon_h, scale = latest_meta
            # the model's (x,y) are in downscaled space; map to absolute screen
            x = int(latest_cmd["x"] / scale)
            y = int(latest_cmd["y"] / scale)
            act({"action": "move_mouse", "x": x, "y": y})
        else:
            act(latest_cmd)

        time.sleep(0.01)


def decide_action(frame_bgr: np.ndarray, meta: Tuple[int, int, float]) -> dict:
    mon_w, mon_h, scale = meta

    try:
        mx, my = pyautogui.position()
    except Exception:
        mx, my = -1, -1

    annotated = frame_bgr.copy()
    down_mouse = (-1, -1)
    if mx >= 0 and my >= 0:
        annotated = annotate_mouse(annotated, mx, my, scale)
        down_mouse = (int(mx * scale), int(my * scale))

    image = to_pil_bgr(annotated)

    system = (
        SYSTEM_INSTRUCTION
        + f"\nScreen width={mon_w}, height={mon_h}, downscale={scale:.4f}."
        + f"\nCanvas width={image.width}, height={image.height}. Mouse (downscaled)={down_mouse}."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Decide one atomic action for THIS frame only."}
        ]}
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
        )

    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    obj = extract_first_json(text)
    if obj is None:
        return {"action": "none"}

    action = obj.get("action")
    if action == "move_mouse" and "x" in obj and "y" in obj:
        return {"action": "move_mouse", "x": int(obj["x"]), "y": int(obj["y"])}
    if action == "click":
        btn = obj.get("button", "left")
        if btn not in ["left", "right", "middle"]:
            btn = "left"
        return {"action": "click", "button": btn}
    return {"action": "none"}


def grab_screen(fps=FRAME_FPS):
    period = 1.0 / fps
    with mss.mss() as sct:
        mon = sct.monitors[1]  # primary monitor
        mon_w, mon_h = mon["width"], mon["height"]
        while True:
            t0 = time.time()
            raw = sct.grab(mon)
            img = np.array(raw)[:, :, :3]  # BGRA -> BGR
            h, w = img.shape[:2]

            scale = 1.0
            if w > DOWNSCALE_WIDTH:
                scale = DOWNSCALE_WIDTH / float(w)
                img = cv2.resize(img, (DOWNSCALE_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

            if not frame_q.full():
                # push both the frame and metadata
                frame_q.put((img, (mon_w, mon_h, scale)))
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))


def annotate_mouse(img: np.ndarray, mx: int, my: int, scale: float) -> np.ndarray:
    # draw a small cross in downscaled space
    dx, dy = int(mx * scale), int(my * scale)
    h, w = img.shape[:2]
    if 0 <= dx < w and 0 <= dy < h:
        cv2.drawMarker(img, (dx, dy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    return img


if __name__ == "__main__":
    print("[boot] starting screen capture & controller…")
    t = threading.Thread(target=grab_screen, daemon=True)
    t.start()
    controller()
