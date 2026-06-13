"""
Phase 2: LoRA fine-tune gemma-3-4b-it to report injected emotional states.

For each example we inject coeff * unit_vector(emotion) at the target layer
(coeff = intensity_fraction * residual_norm) while the model is teacher-forced on
a target that names the emotion at the matching intensity (or reports neutrality
for controls). Only the target tokens contribute to the loss. We train LoRA
adapters; the base weights are frozen.

Usage:
    python train_introspection.py [--epochs 3] [--batch-size 8] [--lr 2e-4]
                                  [--alpha-scale 1.0] [--out adapters/introspection]
Prereqs:
    pip install peft
    python generate_introspection_data.py   # writes data/introspection/train.jsonl
    data/vectors/emotion_vectors_layer{L}.npz from Phase 1
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from emotion_validation.config import MODEL_ID, TARGET_LAYER, VECTORS_DIR
from emotion_validation.inject import (
    estimate_residual_norm,
    inject,
    load_unit_vectors,
    resolve_layer,
)
from emotion_validation.introspection_data import INTENSITY_FRACTIONS, INTROSPECTION_PROMPTS
from emotion_validation.model_utils import load_model_and_tokenizer


class IntrospectionDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.tok = tokenizer
        self.records = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        templated = self.tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, tokenize=True,
        )
        prompt_ids = templated if isinstance(templated, list) else templated["input_ids"]
        target_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tok.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": input_ids,
            "labels": labels,
            "emotion": r["emotion"],
            "intensity": r["intensity"],
        }


def make_collate(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
            "emotion": [b["emotion"] for b in batch],
            "intensity": [b["intensity"] for b in batch],
        }
    return collate


def build_addition(emotions, intensities, unit_vecs, R, alpha_scale, d_model, device):
    """(batch, 1, d_model) per-row injection: coeff * unit(emotion), 0 for controls."""
    add = torch.zeros(len(emotions), d_model, device=device)
    for i, (emo, inten) in enumerate(zip(emotions, intensities)):
        if emo != "none":
            coeff = INTENSITY_FRACTIONS[inten] * R * alpha_scale
            add[i] = coeff * unit_vecs[emo]
    return add.unsqueeze(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--alpha-scale", type=float, default=1.0,
                   help="Global multiplier on all intensity fractions (calibration knob)")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--data", default="data/introspection/train.jsonl")
    p.add_argument("--out", default="adapters/introspection")
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model  # imported here so the dep is optional

    print(f"Loading {MODEL_ID} (bf16) …")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    d_model = model.config.text_config.hidden_size if hasattr(model.config, "text_config") \
        else model.config.hidden_size

    # Injection scale, vectors, and the layer module (grab BEFORE peft-wrapping).
    print("Estimating residual-stream norm at layer", args.layer, "…")
    R = estimate_residual_norm(model, tokenizer, args.layer, INTROSPECTION_PROMPTS[:8], device=device)
    print(f"  residual norm R ≈ {R:.2f}")
    unit_vecs = load_unit_vectors(VECTORS_DIR / f"emotion_vectors_layer{args.layer}.npz",
                                  device=device, dtype=torch.float32)
    layer_module = resolve_layer(model, args.layer)

    # LoRA
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.train()

    ds = IntrospectionDataset(Path(args.data), tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate(tokenizer.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            addition = build_addition(batch["emotion"], batch["intensity"],
                                      unit_vecs, R, args.alpha_scale, d_model, device)
            inputs = {k: batch[k].to(device) for k in ("input_ids", "labels", "attention_mask")}
            with inject(layer_module, addition):
                loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}  loss {loss.item():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    # Stash the injection scale used, so eval matches training.
    with open(Path(args.out) / "inject_meta.json", "w") as f:
        json.dump({"layer": args.layer, "residual_norm": R, "alpha_scale": args.alpha_scale,
                   "intensity_fractions": INTENSITY_FRACTIONS}, f, indent=2)
    print(f"Saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
