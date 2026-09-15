"""GRPO building blocks checked against hand computation (no big model)."""
import math

import torch

from parallax.train.grpo import ADV_EPS, group_advantages
from parallax.train.logprobs import completion_logprobs


class TinyLM(torch.nn.Module):
    """Deterministic causal LM stub: logits depend on the input ids at each position."""

    def __init__(self, vocab=11):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, vocab)
        torch.manual_seed(0)
        self.emb.weight.data = torch.randn(vocab, vocab)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0, **kw):
        logits = self.emb(input_ids)                        # [1, T, V]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return type("Out", (), {"logits": logits})()


def test_completion_logprobs_matches_manual_gather():
    lm = TinyLM()
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
    n = 3                                                   # completion = last 3 tokens
    got = completion_logprobs(lm, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, n)
    full = torch.log_softmax(lm.emb(ids)[0].float(), dim=-1)  # [T, V]
    P = ids.shape[1] - n
    expected = torch.stack([full[P - 1 + i, ids[0, P + i]] for i in range(n)])
    assert torch.allclose(got, expected, atol=1e-6)
    assert got.shape == (n,)


def test_completion_logprobs_temperature():
    lm = TinyLM()
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
    a = completion_logprobs(lm, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, 2, temperature=1.0)
    b = completion_logprobs(lm, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, 2, temperature=2.0)
    assert not torch.allclose(a, b)


def test_group_advantages():
    r = [1.5, 1.0, 0.0, 0.0]
    mean = 0.625
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / 4)
    exp = [(x - mean) / (std + ADV_EPS) for x in r]
    assert group_advantages(r) == exp
    assert group_advantages([0.0, 0.0, 0.0, 0.0]) == [0.0] * 4
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]
    a = group_advantages([1.5, 0.0])
    assert a[0] > 0 > a[1] and abs(a[0] + a[1]) < 1e-9
