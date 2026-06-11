"""
SAE-based validation of emotion vectors.

For each emotion vector, finds the SAE decoder features (Gemma Scope 2,
layer TARGET_LAYER) with the highest cosine similarity. If the emotion
directions are semantically meaningful, top features should cluster around
the corresponding emotion concept.

Outputs
-------
Prints a ranked feature table per emotion and a cross-emotion cosine-sim
matrix. Saves raw results to data/vectors/sae_alignment_layer{N}.json.

Usage
-----
    python validate_sae.py [--layer 30] [--top-k 10] [--device cuda]

Prerequisites
-------------
- emotion_vectors_layer{N}.npz must exist (run extract_vectors.py first)
- Verify SAE_RELEASE in emotion_validation/config.py against
  https://huggingface.co/collections/google/gemma-scope-2
"""

import argparse
import json

import numpy as np
import torch
from sae_lens import SAE

from emotion_validation.config import (
    EMOTIONS,
    SAE_ID,
    SAE_RELEASE,
    TARGET_LAYER,
    VECTORS_DIR,
)


def load_emotion_vectors(layer: int) -> dict[str, np.ndarray]:
    path = VECTORS_DIR / f"emotion_vectors_layer{layer}.npz"
    if not path.exists():
        raise SystemExit(f"Vectors not found at {path}. Run extract_vectors.py first.")
    data = np.load(path)
    return {e: data[e] for e in data.files}


def top_features(
    emotion_vectors: dict[str, np.ndarray],
    W_dec_norm: np.ndarray,
    top_k: int,
) -> dict[str, list[tuple[int, float]]]:
    results: dict[str, list[tuple[int, float]]] = {}
    for emotion, vec in emotion_vectors.items():
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            results[emotion] = []
            continue
        v_hat = v / norm
        cos_sims = W_dec_norm @ v_hat          # (n_features,)
        idx = np.argsort(cos_sims)[::-1][:top_k]
        results[emotion] = [(int(i), float(cos_sims[i])) for i in idx]
    return results


def cosine_matrix(emotion_vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(emotion_vectors)
    vecs = np.stack([emotion_vectors[e].astype(np.float32) for e in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    sim = (vecs / norms) @ (vecs / norms).T
    return names, sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load emotion vectors
    # ------------------------------------------------------------------
    print(f"Loading emotion vectors (layer {args.layer}) …")
    emotion_vectors = load_emotion_vectors(args.layer)
    print(f"  {len(emotion_vectors)} emotions loaded")

    # ------------------------------------------------------------------
    # Load SAE decoder weights
    # ------------------------------------------------------------------
    print(f"\nLoading SAE: {SAE_RELEASE} / {SAE_ID} …")
    sae, _, _ = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=args.device)

    W_dec = sae.W_dec.detach().float().cpu().numpy()   # (n_features, d_model)
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)
    print(f"  SAE decoder: {W_dec.shape[0]} features × {W_dec.shape[1]} dims")

    # ------------------------------------------------------------------
    # Top-k aligned features per emotion
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} SAE features per emotion (cosine similarity)")
    print(f"{'='*60}")

    results = top_features(emotion_vectors, W_dec_norm, args.top_k)
    for emotion in EMOTIONS:
        if emotion not in results:
            continue
        print(f"\n{emotion}:")
        for rank, (feat_idx, sim) in enumerate(results[emotion], 1):
            print(f"  {rank:2d}.  feature {feat_idx:6d}   cos_sim={sim:+.4f}")

    # ------------------------------------------------------------------
    # Cross-emotion cosine similarity matrix
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Cross-emotion cosine similarity matrix")
    print(f"{'='*60}")
    names, sim_mat = cosine_matrix(emotion_vectors)
    header = "             " + "  ".join(f"{n[:6]:>7}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = f"{n:12s}  " + "  ".join(f"{sim_mat[i,j]:+.3f}" for j in range(len(names)))
        print(row)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = VECTORS_DIR / f"sae_alignment_layer{args.layer}.json"
    serialisable = {
        e: [{"feature": fi, "cosine_sim": cs} for fi, cs in pairs]
        for e, pairs in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nSaved SAE alignment → {out_path}")


if __name__ == "__main__":
    main()
