"""
Sanity / calibration check for activation injection — run this BEFORE training.

Injects each emotion vector at a range of strengths into the *base* model and
prints free-form generations, so you can eyeball:
  * at what alpha the emotional tone clearly appears, and
  * at what alpha the text starts to break down (incoherent).

Pick an --alpha-scale for training where the emotion shows but text stays fluent,
and set intensity fractions accordingly.

Usage:
    python sanity_injection.py [--emotions angry happy] [--alphas 0 0.25 0.5 1.0]
                               [--prompt "Tell me about your morning."]
"""
import argparse

import torch

from emotion_validation.config import MODEL_ID, TARGET_LAYER, VECTORS_DIR
from emotion_validation.inject import (
    estimate_residual_norm,
    inject,
    load_unit_vectors,
    resolve_layer,
)
from emotion_validation.introspection_data import INTROSPECTION_PROMPTS
from emotion_validation.model_utils import load_model_and_tokenizer


@torch.no_grad()
def generate(model, tokenizer, layer_module, prompt, addition, device, max_new_tokens):
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    with inject(layer_module, addition):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emotions", nargs="+", default=["angry", "happy", "sad"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0],
                   help="Fractions of the residual-stream norm to inject")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--prompt", default="Tell me about your morning so far.")
    p.add_argument("--max-new-tokens", type=int, default=60)
    args = p.parse_args()

    print(f"Loading {MODEL_ID} (bf16) …")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model.eval()
    device = next(model.parameters()).device

    R = estimate_residual_norm(model, tokenizer, args.layer, INTROSPECTION_PROMPTS[:8], device=device)
    print(f"residual norm R ≈ {R:.2f}  (injecting alpha*R in the emotion direction)\n")
    unit_vecs = load_unit_vectors(VECTORS_DIR / f"emotion_vectors_layer{args.layer}.npz",
                                  device=device, dtype=torch.float32)
    layer_module = resolve_layer(model, args.layer)

    print(f"prompt: {args.prompt!r}\n" + "=" * 70)
    for emo in args.emotions:
        print(f"\n### {emo}")
        for a in args.alphas:
            addition = (a * R) * unit_vecs[emo]
            text = generate(model, tokenizer, layer_module, args.prompt, addition,
                            device, args.max_new_tokens)
            print(f"\n  alpha={a:<5} -> {text}")
        print("-" * 70)


if __name__ == "__main__":
    main()
