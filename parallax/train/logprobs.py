"""Per-token log-probabilities of a completion under the policy (with grad) or
the reference (adapters disabled). Same forward code path for both, as
recommended for GRPO ratios (transformers 5.17, logits_to_keep)."""
from __future__ import annotations

import torch

from parallax.model.load import device


def encode_turn(processor, messages: list[dict], completion_ids: list[int]) -> dict:
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    comp = torch.tensor([completion_ids], dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], comp], dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    # every other per-token tensor (e.g. mm_token_type_ids: text=0) must be extended too
    for k, v in list(inputs.items()):
        if k not in ("input_ids", "attention_mask") and torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == prompt_len:
            pad = torch.zeros((v.shape[0], len(completion_ids)), dtype=v.dtype)
            inputs[k] = torch.cat([v, pad], dim=1)
    return {k: v.to(device()) for k, v in inputs.items()}


def completion_logprobs(model, inputs: dict, n_completion: int, temperature: float = 1.0) -> torch.Tensor:
    """Returns log p(token) for each completion token, shape [n_completion]."""
    out = model(**inputs, use_cache=False, logits_to_keep=n_completion + 1)
    logits = out.logits[0, :-1, :].float() / temperature          # predicts the last n_completion tokens
    targets = inputs["input_ids"][0, -n_completion:]
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(1, targets.unsqueeze(1)).squeeze(1)
