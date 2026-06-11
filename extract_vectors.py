"""
Extract contrastive emotion vectors following Lindsey et al. 2026.

Pipeline
--------
1. Load all emotional stories from data/stories/{emotion}.jsonl
2. For each story, extract the mean residual stream from token 50 onward
   at TARGET_LAYER using a forward-hook on model.model.layers[layer].
3. Emotion vector (raw) = per-emotion mean activation − grand mean across emotions.
4. Fit PCA on neutral dialogue activations; project out the top k directions
   explaining 50% of variance (confound removal).
5. Save final emotion vectors to data/vectors/emotion_vectors_layer{N}.npz

Intermediate activations are cached as .npy files in data/vectors/cache/
so re-runs skip already-extracted texts.

Usage
-----
    python extract_vectors.py [--layer 30] [--token-offset 50] [--device cuda]
                              [--emotions excited happy ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from emotion_validation.config import (
    EMOTIONS,
    MODEL_ID,
    NEUTRAL_DIR,
    TARGET_LAYER,
    TOKEN_OFFSET,
    VECTORS_DIR,
    STORIES_DIR,
)
from emotion_validation.model_utils import get_story_activation, load_model_and_tokenizer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_texts(jsonl_path: Path) -> list[str]:
    texts: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            texts.extend(record["items"])
    return texts


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def _filter_finite(acts: np.ndarray, desc: str = "") -> np.ndarray:
    mask = np.isfinite(acts).all(axis=1)
    n_bad = (~mask).sum()
    if n_bad:
        tqdm.write(f"  [{desc}] dropped {n_bad} non-finite vectors")
    return acts[mask]


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    token_offset: int,
    device: str,
    desc: str = "",
) -> np.ndarray:
    acts: list[np.ndarray] = []
    skipped = 0
    for text in tqdm(texts, desc=desc, leave=False):
        act = get_story_activation(model, tokenizer, text, layer, token_offset, device)
        if act is not None:
            acts.append(act.numpy())
        else:
            skipped += 1
    if skipped:
        tqdm.write(f"  [{desc}] skipped {skipped} texts shorter than {token_offset} tokens")
    arr = np.array(acts, dtype=np.float32)
    return _filter_finite(arr, desc)


# ---------------------------------------------------------------------------
# Vector computation
# ---------------------------------------------------------------------------

def compute_emotion_vectors(
    emotion_acts: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Returns (raw_emotion_vectors, grand_mean).
    raw_emotion_vectors[e] = per_emotion_mean[e] - grand_mean
    grand_mean is the mean of per-emotion means (equal weight per emotion).
    """
    per_emotion_mean = {e: acts.mean(axis=0) for e, acts in emotion_acts.items()}
    grand_mean = np.stack(list(per_emotion_mean.values())).mean(axis=0)
    raw_vectors = {e: mean - grand_mean for e, mean in per_emotion_mean.items()}
    return raw_vectors, grand_mean


def compute_confound_directions(neutral_acts: np.ndarray) -> np.ndarray:
    """
    Fit PCA on neutral activations; return the top k components that
    together explain ≥50% of variance. Shape: (k, d_model).
    """
    pca = PCA()
    pca.fit(neutral_acts)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.50)) + 1
    print(f"  {k} PCs explain ≥50% of neutral-dialogue variance")
    return pca.components_[:k].astype(np.float64)


def project_out(vector: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Remove each confound direction from vector via successive projection."""
    v = vector.astype(np.float64)
    for d in directions:
        v = v - np.dot(v, d) * d
    return v.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--token-offset", type=int, default=TOKEN_OFFSET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--emotions", nargs="+", default=EMOTIONS,
                        help="Subset of emotions to process (default: all)")
    args = parser.parse_args()

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = VECTORS_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"Loading {MODEL_ID} …")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, args.device)

    # ------------------------------------------------------------------
    # 1. Extract emotional story activations (with caching)
    # ------------------------------------------------------------------
    print("\nExtracting emotional story activations …")
    emotion_acts: dict[str, np.ndarray] = {}

    for emotion in args.emotions:
        cache_path = cache_dir / f"{emotion}_layer{args.layer}.npy"
        if cache_path.exists():
            emotion_acts[emotion] = _filter_finite(np.load(cache_path), emotion)
            print(f"  {emotion}: {len(emotion_acts[emotion])} vectors (cached)")
            continue

        story_path = STORIES_DIR / f"{emotion}.jsonl"
        if not story_path.exists():
            print(f"  [skip] {emotion}: {story_path} not found — run emotion_stories_generation.py first")
            continue

        texts = load_texts(story_path)
        acts = extract_activations(
            model, tokenizer, texts, args.layer, args.token_offset, args.device,
            desc=emotion,
        )
        np.save(cache_path, acts)
        emotion_acts[emotion] = acts
        print(f"  {emotion}: {len(acts)} vectors extracted")

    if not emotion_acts:
        raise SystemExit("No emotion activations found. Generate stories first.")

    # ------------------------------------------------------------------
    # 2. Compute raw emotion vectors
    # ------------------------------------------------------------------
    print("\nComputing raw emotion vectors …")
    raw_vectors, grand_mean = compute_emotion_vectors(emotion_acts)

    # ------------------------------------------------------------------
    # 3. Extract neutral activations for confound projection (with caching)
    # ------------------------------------------------------------------
    neutral_cache = cache_dir / f"neutral_layer{args.layer}.npy"
    if neutral_cache.exists():
        neutral_acts = _filter_finite(np.load(neutral_cache), "neutral")
        print(f"\nNeutral: {len(neutral_acts)} vectors (cached)")
    else:
        neutral_path = NEUTRAL_DIR / "neutral.jsonl"
        if not neutral_path.exists():
            raise SystemExit(f"Neutral dialogues not found at {neutral_path}. "
                             "Run emotion_stories_generation.py --mode neutral first.")
        print("\nExtracting neutral dialogue activations …")
        texts = load_texts(neutral_path)
        neutral_acts = extract_activations(
            model, tokenizer, texts, args.layer, args.token_offset, args.device,
            desc="neutral",
        )
        np.save(neutral_cache, neutral_acts)
        print(f"  {len(neutral_acts)} neutral vectors extracted")

    # ------------------------------------------------------------------
    # 4. Confound projection
    # ------------------------------------------------------------------
    print("\nFitting confound PCA …")
    confound_dirs = compute_confound_directions(neutral_acts)

    emotion_vectors = {
        e: project_out(raw_vectors[e], confound_dirs)
        for e in raw_vectors
    }

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_path = VECTORS_DIR / f"emotion_vectors_layer{args.layer}.npz"
    np.savez(out_path, **emotion_vectors)
    np.save(VECTORS_DIR / f"grand_mean_layer{args.layer}.npy", grand_mean)
    print(f"\nSaved emotion vectors → {out_path}")

    # Quick diagnostics
    print("\nVector L2 norms:")
    for e, v in emotion_vectors.items():
        print(f"  {e:12s}  {np.linalg.norm(v):.4f}")

    print("\nCross-emotion cosine similarity matrix:")
    names = list(emotion_vectors)
    vecs = np.stack([emotion_vectors[e] for e in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    sim = (vecs / norms) @ (vecs / norms).T
    header = "             " + "  ".join(f"{n[:6]:>6}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = f"{n:12s}  " + "  ".join(f"{sim[i,j]:+.3f}" for j in range(len(names)))
        print(row)


if __name__ == "__main__":
    main()
