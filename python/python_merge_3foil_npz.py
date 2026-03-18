#!/usr/bin/env python3
"""
Merge three SHMS foil .npz datasets into one file.

Usage
-----
python merge_3foil_npz.py <up_npz> <mid_npz> <down_npz> <output_npz> [--weights w_up w_mid w_down] [--seed N]

Examples
--------
python merge_3foil_npz.py \
  ../worksim/shms_3foil_nosieve_up.npz \
  ../worksim/shms_3foil_nosieve_mid.npz \
  ../worksim/shms_3foil_nosieve_down.npz \
  ../worksim/shms_3foil_nosieve_merged.npz

python merge_3foil_npz.py \
  ../worksim/up.npz ../worksim/mid.npz ../worksim/down.npz ../worksim/merged.npz \
  --weights 1 2 1 --seed 42

Notes
-----
- This script expects each input .npz to contain the same keys/arrays.
- Arrays are merged along axis=0.
- If --weights are given, downsampling is applied so merged sample counts follow
  approximately the requested ratio without oversampling.
"""

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np


def _is_mergeable_array(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim >= 1


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load NPZ: {path}\n{exc}") from exc

    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _check_same_keys(ds: List[Dict[str, np.ndarray]]) -> List[str]:
    base = set(ds[0].keys())
    for i, d in enumerate(ds[1:], start=1):
        ks = set(d.keys())
        if ks != base:
            missing = sorted(base - ks)
            extra = sorted(ks - base)
            raise ValueError(
                f"Input #{i+1} keys mismatch.\n"
                f"Missing keys: {missing}\n"
                f"Extra keys: {extra}"
            )
    return sorted(base)


def _infer_event_count(d: Dict[str, np.ndarray], path: str) -> int:
    """
    Infer number of events from event-level arrays only.

    We ignore metadata keys starting with 'merge_' to avoid mixing
    provenance arrays (e.g. merge_sources length=3) with event arrays.
    """
    candidates = []
    for k, v in d.items():
        if not _is_mergeable_array(v):
            continue
        if k.startswith("merge_"):   # <- ignore metadata arrays
            continue
        candidates.append((k, v.shape[0]))

    if not candidates:
        raise ValueError(f"No mergeable event arrays (ndim>=1) found in {path}")

    ref_k, ref_n = candidates[0]
    for k, n in candidates[1:]:
        if n != ref_n:
            raise ValueError(
                f"Inconsistent event dimension in {path}: "
                f"{ref_k}.shape[0]={ref_n}, {k}.shape[0]={n}"
            )
    return ref_n


def _weighted_indices(
    n_up: int,
    n_mid: int,
    n_down: int,
    weights: Tuple[float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.array(weights, dtype=float)
    if np.any(w <= 0):
        raise ValueError("All weights must be > 0")
    w = w / w.sum()

    counts = np.array([n_up, n_mid, n_down], dtype=int)
    # determine max total without oversampling:
    # target_i = total * w_i <= counts_i  => total <= counts_i / w_i
    total_max = int(np.floor(np.min(counts / w)))
    if total_max <= 0:
        raise ValueError("Unable to allocate weighted sample without oversampling.")

    target = np.floor(total_max * w).astype(int)
    # fix rounding leftovers
    rem = total_max - target.sum()
    if rem > 0:
        order = np.argsort(-(total_max * w - target))  # largest fractional part first
        for i in order[:rem]:
            target[i] += 1

    idx_up = rng.choice(n_up, size=target[0], replace=False)
    idx_mid = rng.choice(n_mid, size=target[1], replace=False)
    idx_down = rng.choice(n_down, size=target[2], replace=False)
    return idx_up, idx_mid, idx_down


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge three foil NPZ files.")
    parser.add_argument("up_npz")
    parser.add_argument("mid_npz")
    parser.add_argument("down_npz")
    parser.add_argument("output_npz")
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        metavar=("W_UP", "W_MID", "W_DOWN"),
        help="Optional relative weights for up/mid/down (downsample only, no oversampling).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    paths = [args.up_npz, args.mid_npz, args.down_npz]
    labels = ["up", "mid", "down"]

    ds = [_load_npz(p) for p in paths]
    keys = _check_same_keys(ds)

    counts = [_infer_event_count(d, p) for d, p in zip(ds, paths)]
    print(f"Input event counts: up={counts[0]}, mid={counts[1]}, down={counts[2]}")

    rng = np.random.default_rng(args.seed)

    if args.weights is None:
        idxs = [None, None, None]
        print("Merging with full statistics (simple concatenation).")
    else:
        idxs = _weighted_indices(counts[0], counts[1], counts[2], tuple(args.weights), rng)
        print(
            "Applying weighted downsampling with weights "
            f"{tuple(args.weights)} -> selected: "
            f"up={len(idxs[0])}, mid={len(idxs[1])}, down={len(idxs[2])}"
        )

    merged = {}
    for k in keys:
        a_up, a_mid, a_down = ds[0][k], ds[1][k], ds[2][k]

        # Scalars / metadata must match exactly
        if not _is_mergeable_array(a_up):
            if not (np.array_equal(a_up, a_mid) and np.array_equal(a_up, a_down)):
                raise ValueError(f"Scalar/meta key '{k}' differs across files; cannot merge safely.")
            merged[k] = a_up
            continue

        if args.weights is None:
            merged[k] = np.concatenate([a_up, a_mid, a_down], axis=0)
        else:
            iu, im, idn = idxs
            merged[k] = np.concatenate([a_up[iu], a_mid[im], a_down[idn]], axis=0)

    # Add provenance info
    merged["merge_sources"] = np.array(paths, dtype="<U512")
    merged["merge_labels"] = np.array(labels, dtype="<U16")
    if args.weights is not None:
        merged["merge_weights_requested"] = np.array(args.weights, dtype=float)

    np.savez_compressed(args.output_npz, **merged)

    n_out = _infer_event_count(merged, args.output_npz)
    print(f"Merged file written: {args.output_npz}")
    print(f"Output event count: {n_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())