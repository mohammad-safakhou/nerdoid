from __future__ import annotations

from typing import Optional, Sequence

from .actuator import Actuator
from .capture import ScreenCapturer
from .controller import Controller
from .policy import VisionAgent
from .settings import RuntimeConfig, create_config


def run(task: Optional[str] = None) -> None:
    config = create_config(task_prompt=task)
    _run_with_config(config)


def run_with_args(args: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Nerdoid UI agent.")
    parser.add_argument("--task", type=str, help="High-level goal instruction for the agent.")
    parser.add_argument("--prefer-mps", action="store_true", help="Force-enable MPS when available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS is available.")
    parsed = parser.parse_args(args=args)

    prefer_mps = None
    if parsed.cpu:
        prefer_mps = False
    elif parsed.prefer_mps:
        prefer_mps = True

    config = create_config(task_prompt=parsed.task, prefer_mps=prefer_mps)
    _run_with_config(config)


def _run_with_config(config: RuntimeConfig) -> None:
    print(f"[boot] loading {config.model_id} on {config.device} ({config.dtype}) ...")
    capture = ScreenCapturer(config.frame_fps, config.downscale_width)
    policy = VisionAgent(config)
    actuator = Actuator(config.click_min_period)
    controller = Controller(capture, policy, actuator, config)

    print("[boot] starting screen capture & controller…")
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[shutdown] interrupted by user")


if __name__ == "__main__":
    run()
