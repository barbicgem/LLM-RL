"""
Build the Phase 2 introspection dataset.

Writes:
  data/introspection/train.jsonl  — TRAIN_EMOTIONS x train-prompts x intensities + controls
  data/introspection/eval.jsonl   — (TRAIN + HELD_OUT emotions) x eval-prompts x intensities + controls

Eval records carry "held_out": bool so the eval script can report generalization
accuracy on emotions the model was never trained to introspect.

Usage:
    python generate_introspection_data.py [--seed 0] [--eval-frac 0.2]
"""
import argparse
import json
import random
from pathlib import Path

from emotion_validation.introspection_data import (
    HELD_OUT_EMOTIONS,
    TRAIN_EMOTIONS,
    build_split,
    split_prompts,
)

OUT_DIR = Path("data/introspection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_prompts, eval_prompts = split_prompts(rng, args.eval_frac)

    # Train: only the 8 train emotions, only train prompts.
    train = build_split(TRAIN_EMOTIONS, train_prompts, rng)

    # Eval: all 12 emotions, only held-out prompts. Tag held-out emotions.
    eval_records = build_split(TRAIN_EMOTIONS + HELD_OUT_EMOTIONS, eval_prompts, rng)
    held = set(HELD_OUT_EMOTIONS)
    for r in eval_records:
        r["held_out"] = r["emotion"] in held

    with open(OUT_DIR / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "eval.jsonl", "w") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    n_ctrl = sum(1 for r in train if r["emotion"] == "none")
    print(f"train: {len(train)} records ({n_ctrl} controls, {len(train)-n_ctrl} injected)")
    print(f"  train emotions: {TRAIN_EMOTIONS}")
    print(f"  train prompts:  {len(train_prompts)}")
    print(f"eval:  {len(eval_records)} records  (held-out emotions: {HELD_OUT_EMOTIONS})")
    print(f"  eval prompts:   {len(eval_prompts)}")
    print(f"Wrote -> {OUT_DIR}/train.jsonl, {OUT_DIR}/eval.jsonl")


if __name__ == "__main__":
    main()
