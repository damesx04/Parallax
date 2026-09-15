"""Model: model identity, LoRA targets, adapter round trip, visual token grid (slow: loads the model)."""
import pytest
import torch
from PIL import Image

from parallax.model.load import (MAX_PIXELS, MIN_PIXELS, MODEL_ID, attach_lora, image_token_count,
                                 load_base_model, load_processor)


def test_model_id_is_fixed():
    assert MODEL_ID == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_processor_grid():
    proc = load_processor()
    assert image_token_count(proc, Image.new("RGB", (896, 672), "white")) == 32 * 24 == 768
    assert image_token_count(proc, Image.new("RGB", (224, 168), "white")) == 8 * 6 == 48
    ip = proc.image_processor
    assert ip.size["shortest_edge"] == MIN_PIXELS and ip.size["longest_edge"] == MAX_PIXELS


@pytest.fixture(scope="module")
def model():
    return load_base_model()


def test_lora_targets_only_language_model(model, tmp_path):
    peft_model = attach_lora(model)
    lora_modules = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
    assert lora_modules, "no LoRA layers attached"
    assert all("language_model" in n for n in lora_modules)
    assert not any("visual" in n for n in lora_modules)
    assert len(lora_modules) == 36 * 7
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    assert 0 < trainable / total < 0.02
    # adapter round trip
    peft_model.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    from safetensors.torch import load_file
    sd = load_file(tmp_path / "adapter" / "adapter_model.safetensors")
    assert all("language_model" in k for k in sd)
    base = peft_model.unload()
    reloaded = attach_lora(base, str(tmp_path / "adapter"), trainable=False)
    assert sum(1 for n, m in reloaded.named_modules() if hasattr(m, "lora_A")) == 36 * 7
    reloaded.unload()


def test_encode_turn_extends_every_per_token_tensor():
    from parallax.train.logprobs import encode_turn
    proc = load_processor()
    msgs = [{"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (224, 168), "white")}, {"type": "text", "text": "hi"}]}]
    comp = [1, 2, 3, 4, 5]
    enc = encode_turn(proc, msgs, comp)
    n = enc["input_ids"].shape[1]
    assert enc["input_ids"][0, -5:].tolist() == comp
    for k, v in enc.items():
        if torch.is_tensor(v) and v.dim() == 2 and k != "pixel_values" and k != "image_grid_thw":
            assert v.shape[1] == n, (k, v.shape)
    assert "mm_token_type_ids" in enc and enc["mm_token_type_ids"][0, -5:].sum().item() == 0


def test_training_prep_keeps_only_lora_trainable(model):
    from parallax.train.grpo import prepare_for_training
    peft_model = attach_lora(model)
    params = prepare_for_training(peft_model)
    by_id = {id(p): n for n, p in peft_model.named_parameters()}
    names = [by_id[id(p)] for p in params]
    assert len(names) == 36 * 7 * 2
    assert all("lora_" in n and "language_model" in n for n in names)
    assert not any("visual" in n for n in names)
    assert not any(p.requires_grad for n, p in peft_model.named_parameters() if "lora_" not in n)
    peft_model.unload()
