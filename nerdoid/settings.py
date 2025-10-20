from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


@dataclass
class RuntimeConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    frame_fps: int = 10
    decision_hz: int = 4
    max_new_tokens: int = 96
    downscale_width: int = 1280
    click_min_period: float = 0.3
    queue_maxsize: int = 2
    max_action_history: int = 6
    task_prompt: str | None = field(default_factory=lambda: os.environ.get("NERDOID_TASK") or None)
    prefer_mps: bool = field(default_factory=lambda: os.environ.get("NERDOID_USE_MPS", "0") == "1")
    device: str = field(init=False)
    dtype: torch.dtype = field(init=False)

    def __post_init__(self) -> None:
        if self.prefer_mps and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

    @property
    def decision_period(self) -> float:
        return 1.0 / float(self.decision_hz)


def create_config(**overrides) -> RuntimeConfig:
    sanitized = {k: v for k, v in overrides.items() if v is not None}
    return RuntimeConfig(**sanitized)
