"""
Activation injection for Phase 2 (injection-guided transparency training).

Adds ``coeff * unit_emotion_vector`` to the residual stream at a target layer via
a forward hook, so the model "experiences" a known emotional state. Because we
control what is injected, the injected emotion is ground-truth supervision for
the model's self-report.

The same hook works during training (one teacher-forced forward over the whole
sequence) and during generation (it fires on every decode step), because the
addition is broadcast over the sequence dimension.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from emotion_validation.model_utils import _get_layer


def resolve_layer(model, layer: int):
    """Return the decoder-layer nn.Module, unwrapping a PEFT/LoRA wrapper if present.

    Grab this BEFORE wrapping with get_peft_model, or pass the base model — the
    returned module object stays valid after wrapping, so the injection hook keeps
    firing regardless of the PEFT wrapper.
    """
    try:
        return _get_layer(model, layer)
    except AttributeError:
        # PeftModel -> .base_model.model is the original model
        base = getattr(getattr(model, "base_model", None), "model", None)
        if base is not None:
            return _get_layer(base, layer)
        raise


def load_unit_vectors(npz_path, device="cuda", dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Load Phase 1 emotion vectors; return {emotion: unit-norm tensor (d_model,)}."""
    data = np.load(npz_path)
    vecs: dict[str, torch.Tensor] = {}
    for e in data.files:
        v = torch.tensor(np.asarray(data[e]), dtype=dtype, device=device)
        vecs[e] = v / (v.norm() + 1e-8)
    return vecs


@torch.no_grad()
def estimate_residual_norm(model, tokenizer, layer, texts, skip_tokens=1, device="cuda") -> float:
    """Median per-token L2 norm of the residual stream at ``layer`` over ``texts``.

    Injection strength is expressed as a fraction of this, so the perturbation is
    scaled to the model's natural activation magnitude rather than an arbitrary
    absolute number.

    We skip the first ``skip_tokens`` positions (the BOS token carries a massive
    outlier activation in Gemma-3 that would inflate the estimate) and use the
    median, which is robust to any remaining attention-sink / massive-activation
    tokens.
    """
    module = resolve_layer(model, layer)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = module.register_forward_hook(hook)
    norms: list[float] = []
    try:
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
            h = captured["h"][0, skip_tokens:, :].float()       # (seq', d_model)
            if h.shape[0] == 0:
                continue
            norms.extend(h.norm(dim=-1).tolist())
    finally:
        handle.remove()
    return float(np.median(norms)) if norms else 0.0


@contextmanager
def inject(layer_module, addition: torch.Tensor):
    """Add ``addition`` to the output of ``layer_module`` for every forward pass
    while active.

    ``addition`` must broadcast to (batch, seq, d_model). Use shape:
      * (d_model,)        — same vector for the whole batch
      * (batch, 1, d_model) — per-row vectors (for mixed-emotion training batches)
    """
    def hook(_m, _i, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        hs = hs + addition.to(hs.dtype).to(hs.device)
        if is_tuple:
            return (hs, *output[1:])
        return hs

    handle = layer_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
