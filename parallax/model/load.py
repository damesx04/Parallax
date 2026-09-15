"""Model/processor loading for Qwen/Qwen2.5-VL-3B-Instruct on MPS (R8).

Verified against transformers 5.17 source: class Qwen2_5_VLForConditionalGeneration,
`dtype=` kwarg, sdpa attention; processor min/max_pixels map to the image
processor's shortest/longest edge so 896x672 and 224x168 are never resized.
"""
from __future__ import annotations

import functools
import os

import torch

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SCREEN_W, SCREEN_H = 896, 672
CROP_W, CROP_H = 224, 168
MIN_PIXELS = CROP_W * CROP_H
MAX_PIXELS = SCREEN_W * SCREEN_H
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@functools.lru_cache(maxsize=1)
def load_processor():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


def load_base_model(dtype=torch.bfloat16):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa")
    model.to(device())
    model.eval()
    return model


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET_REGEX,
                      task_type="CAUSAL_LM")


def attach_lora(model, adapter_dir: str | None = None, trainable: bool = True):
    """Wraps the base model with LoRA (fresh) or loads an adapter directory."""
    from peft import PeftModel, get_peft_model
    if adapter_dir:
        return PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    return get_peft_model(model, lora_config())


def image_token_count(processor, image) -> int:
    """Number of visual tokens the processor produces for a PIL image."""
    out = processor.image_processor(images=[image], return_tensors="pt")
    thw = out["image_grid_thw"][0]
    merge = processor.image_processor.merge_size
    return int(thw[0] * thw[1] * thw[2]) // (merge * merge)
