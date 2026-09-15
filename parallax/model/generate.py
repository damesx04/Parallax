"""Text generation wrapper around Qwen2.5-VL (shared by all policies)."""
from __future__ import annotations

import base64
import io

import torch
from PIL import Image

from parallax.model.load import device
from parallax.obs import render_user_text

MAX_NEW_TOKENS = 64


def png_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def build_messages(obs: dict, system_prompt: str, with_image: bool, extra_instruction: str | None = None) -> list[dict]:
    """One shared prompt template; DOM-only differs only by omitting images."""
    user_text = render_user_text(obs["task"], obs["tree_text"], obs["history"], obs["step"], obs["max_steps"])
    if extra_instruction:
        user_text = user_text + "\n" + extra_instruction
    content = []
    if with_image:
        content.append({"type": "image", "image": png_to_image(obs["screenshot_png"])})
        if obs.get("verify_crop_png"):
            content.append({"type": "text", "text": "Zoomed crop around the element you just verified:"})
            content.append({"type": "image", "image": png_to_image(obs["verify_crop_png"])})
    content.append({"type": "text", "text": user_text})
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content}]


class Generator:
    """Holds the model + processor; produces one completion per call."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def encode(self, messages: list[dict]) -> dict:
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt")
        return {k: v.to(device()) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, messages: list[dict], sample: bool, temperature: float = 1.0,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
        inputs = self.encode(messages)
        # explicit sampling parameters: the model card's generation_config has top_k=1/top_p=0.001,
        # which would silently make "sampling" greedy
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=sample, pad_token_id=self.processor.tokenizer.pad_token_id,
                      repetition_penalty=1.0)
        if sample:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = out[0, prompt_len:].tolist()
        text = self.processor.decode(completion_ids, skip_special_tokens=True)
        return {"text": text, "completion_ids": completion_ids, "prompt_len": prompt_len,
                "messages": messages}
