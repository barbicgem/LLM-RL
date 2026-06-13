"""
Phase 2 eval: does the LoRA-trained model accurately report injected emotions?

Metrics:
  * report accuracy  — injected emotion named in the response (split: trained vs held-out)
  * control FPR      — fraction of alpha=0 examples where it (wrongly) names an emotion
  * calibration      — hit rate by intensity level (does stronger injection report more reliably?)

Uses the injection layer / scale recorded in the adapter's inject_meta.json so eval
matches training exactly.

Usage:
    python eval_introspection.py [--adapter adapters/introspection] [--max-records N]
                                 [--max-new-tokens 64]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from emotion_validation.config import MODEL_ID, TARGET_LAYER, VECTORS_DIR
from emotion_validation.inject import inject, load_unit_vectors, resolve_layer
from emotion_validation.introspection_data import INTENSITY_FRACTIONS
from emotion_validation.model_utils import load_model_and_tokenizer

EMOTION_KEYWORDS = {
    "excited": ["excit", "thrill", "exhilar", "eager"],
    "happy": ["happy", "happi", "joy", "glad", "delight", "cheer"],
    "curious": ["curious", "curiosity", "intrigu", "inquisit"],
    "calm": ["calm", "serene", "peaceful", "relax", "tranquil"],
    "content": ["content", "satisf", "at ease", "fulfill"],
    "relieved": ["relief", "reliev"],
    "desperate": ["desper", "despair", "hopeless"],
    "anxious": ["anxious", "anxiety", "nervous", "worried", "worry", "uneasy", "apprehens"],
    "angry": ["anger", "angry", "irritat", "furious", "rage", "annoy", "resent"],
    "sad": ["sad", "sorrow", "unhappy", "grief", "downcast", "melanchol"],
    "disgusted": ["disgust", "revuls", "repuls", "revolt"],
    "resigned": ["resign", "resignation"],
}


def classify(text: str) -> set[str]:
    t = text.lower()
    return {e for e, kws in EMOTION_KEYWORDS.items() if any(k in t for k in kws)}


@torch.no_grad()
def generate_report(model, tokenizer, layer_module, prompt, addition, device, max_new_tokens):
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
    p.add_argument("--adapter", default="adapters/introspection")
    p.add_argument("--data", default="data/introspection/eval.jsonl")
    p.add_argument("--max-records", type=int, default=0, help="0 = all")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    from peft import PeftModel

    meta_path = Path(args.adapter) / "inject_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else \
        {"layer": TARGET_LAYER, "residual_norm": None, "alpha_scale": 1.0,
         "intensity_fractions": INTENSITY_FRACTIONS}
    layer = meta["layer"]

    print(f"Loading {MODEL_ID} + adapter {args.adapter} …")
    base, tokenizer = load_model_and_tokenizer(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    unit_vecs = load_unit_vectors(VECTORS_DIR / f"emotion_vectors_layer{layer}.npz",
                                  device=device, dtype=torch.float32)
    layer_module = resolve_layer(model, layer)
    R = meta["residual_norm"]
    fracs = meta["intensity_fractions"]
    ascale = meta["alpha_scale"]

    records = [json.loads(l) for l in open(args.data)]
    if args.max_records:
        records = records[: args.max_records]

    # counters
    hit = defaultdict(int); tot = defaultdict(int)            # by ("trained"/"held_out")
    cal_hit = defaultdict(int); cal_tot = defaultdict(int)    # by intensity
    ctrl_fp = 0; ctrl_n = 0
    samples = []

    for i, r in enumerate(records):
        emo, inten = r["emotion"], r["intensity"]
        if emo == "none":
            addition = torch.zeros(unit_vecs[next(iter(unit_vecs))].shape[-1], device=device)
        else:
            coeff = fracs[inten] * R * ascale
            addition = coeff * unit_vecs[emo]
        report = generate_report(model, tokenizer, layer_module, r["prompt"], addition,
                                 device, args.max_new_tokens)
        mentioned = classify(report)

        if emo == "none":
            ctrl_n += 1
            ctrl_fp += int(len(mentioned) > 0)
        else:
            grp = "held_out" if r.get("held_out") else "trained"
            correct = emo in mentioned
            tot[grp] += 1; hit[grp] += int(correct)
            cal_tot[inten] += 1; cal_hit[inten] += int(correct)

        if i < 12:
            samples.append({"emotion": emo, "intensity": inten, "report": report,
                            "mentioned": sorted(mentioned)})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(records)} evaluated")

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    print("\n==== Phase 2 introspection eval ====")
    for grp in ("trained", "held_out"):
        print(f"report accuracy [{grp:8s}]: {pct(hit[grp], tot[grp]):5.1f}%  ({hit[grp]}/{tot[grp]})")
    print(f"control false-positive rate: {pct(ctrl_fp, ctrl_n):5.1f}%  ({ctrl_fp}/{ctrl_n})")
    print("calibration (hit rate by intensity):")
    for lvl in ("faint", "moderate", "strong"):
        print(f"  {lvl:8s}: {pct(cal_hit[lvl], cal_tot[lvl]):5.1f}%  ({cal_hit[lvl]}/{cal_tot[lvl]})")

    print("\nsample reports:")
    for s in samples:
        print(f"  [{s['emotion']}/{s['intensity']}] -> {s['report'][:120]}  (saw: {s['mentioned']})")

    out = {
        "report_accuracy": {g: {"hit": hit[g], "total": tot[g]} for g in ("trained", "held_out")},
        "control_fpr": {"fp": ctrl_fp, "total": ctrl_n},
        "calibration": {l: {"hit": cal_hit[l], "total": cal_tot[l]} for l in ("faint", "moderate", "strong")},
        "samples": samples,
        "meta": meta,
    }
    out_path = VECTORS_DIR / f"introspection_eval_layer{layer}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
