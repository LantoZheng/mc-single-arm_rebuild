#!/usr/bin/env python3
"""
Convert mc-single-arm SHMS `.npz` output into the feature/target layout
expected by SHMS_Calibration_NN.

Inputs
------
- mc-single-arm npz with key `ntuple` (SHMS layout, 23 columns).
- Optional `.inp` file to auto-infer p_set (GeV/c) and spectrometer angle.

Output
------
Compressed `.npz` with:
    inputs  : shape (N, 7) columns [x_fp, y_fp, xp_fp, yp_fp, x_tar, p_set, l_mag]
    targets : shape (N, 4) columns [delta, y_tar, xptar, yptar]
    input_features  : string labels for inputs
    target_features : string labels for targets
    meta_json       : JSON string with provenance (source paths, p0, angle, l_mag mode)

Usage
-----
python postprocess_mc_to_shms_nn.py ../worksim/shms_nn_train_3gev15deg.npz
    --input-config ../infiles/shms_nn_train_3gev15deg.inp
    --output ../worksim/shms_nn_train_3gev15deg_nn_ready.npz
    --lmag-mode straight --lmag-baseline-cm 308.0

Notes
-----
- By default failed events (stop_id != 0) are dropped; pass --keep-failed to retain.
- p_set (GeV/c) is taken from --p0, otherwise inferred from the `.inp` file.
- l_mag can be a constant zero, a straight-line path-length estimate, or an
  existing ntuple column (see --lmag-mode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Ntuple column indices (SHMS layout) ─────────────────────────────────────────
IDX_X_FP = 0
IDX_Y_FP = 1
IDX_XP_FP = 2
IDX_YP_FP = 3
IDX_ZTAR_INIT = 4   # used as y_tar target
IDX_YTAR_INIT = 5
IDX_DPP_INIT = 6
IDX_DTH_INIT = 7    # generated in mrad, stored here in radians
IDX_DPH_INIT = 8
IDX_XTAR_INIT = 14
IDX_STOP_ID = 20
# Other ntuple columns (reconstruction outputs, sieve placeholders, beam coords)
# are not needed for the NN alignment and are intentionally omitted here.


def _default_config_from_npz(npz_path: Path) -> Optional[Path]:
    """
    Heuristic: walk up the directory tree and look for infiles/<stem>.inp.

    Returns the first matching Path if found, otherwise None. ``stem`` is the
    NPZ filename without the extension.
    """
    stem = npz_path.stem
    for parent in npz_path.parents:
        candidate = parent / "infiles" / f"{stem}.inp"
        if candidate.exists():
            return candidate.resolve()
    return None


def _parse_numeric_lines(path: Path) -> List[float]:
    """
    Read numeric tokens from an .inp file, stripping comments.
    """
    values: List[float] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()[0]
        try:
            values.append(float(tok))
        except ValueError:
            continue
    return values


def _load_inp_metadata(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (p_spec_gev, theta_deg) if they can be read, else (None, None).
    """
    try:
        vals = _parse_numeric_lines(path)
    except FileNotFoundError:
        return None, None

    if len(vals) < 4:
        return None, None
    p_spec_mev = vals[2]  # value stored in the .inp file (MeV/c)
    theta_deg = vals[3]
    return p_spec_mev / 1000.0, theta_deg


def _compute_lmag(
    xp_fp: np.ndarray,
    yp_fp: np.ndarray,
    base_cm: float,
    mode: str,
    ntuple: np.ndarray,
    column: Optional[int],
) -> np.ndarray:
    if mode == "zero":
        return np.zeros_like(xp_fp, dtype=np.float32)
    if mode == "straight":
        # Straight-line estimate to a focal-plane distance `base_cm`
        path_length_factor = np.sqrt(1.0 + np.square(xp_fp) + np.square(yp_fp))
        path_length_cm = path_length_factor * np.float32(base_cm)
        return path_length_cm.astype(np.float32)
    if mode == "column":
        if column is None:
            raise ValueError("--lmag-column must be provided when mode='column'")
        if column >= ntuple.shape[1]:
            raise ValueError(f"Requested l_mag column {column} exceeds ntuple width {ntuple.shape[1]}")
        return ntuple[:, column].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported l_mag mode: {mode}")


def _build_output_arrays(
    ntuple: np.ndarray,
    p0_gev: float,
    lmag_mode: str,
    lmag_baseline_cm: float,
    lmag_column: Optional[int],
    keep_failed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ntuple.shape[1] <= IDX_XTAR_INIT:
        raise ValueError(f"Expected >= {IDX_XTAR_INIT + 1} columns for SHMS ntuple, got {ntuple.shape[1]}")

    if not keep_failed:
        if ntuple.shape[1] <= IDX_STOP_ID:
            raise ValueError("stop_id column missing; cannot filter failed events.")
        mask = ntuple[:, IDX_STOP_ID] == 0
        ntuple = ntuple[mask]

    if ntuple.size == 0:
        raise ValueError("No events remain after filtering.")

    x_fp = ntuple[:, IDX_X_FP].astype(np.float32, copy=False)
    y_fp = ntuple[:, IDX_Y_FP].astype(np.float32, copy=False)
    xp_fp = ntuple[:, IDX_XP_FP].astype(np.float32, copy=False)
    yp_fp = ntuple[:, IDX_YP_FP].astype(np.float32, copy=False)
    x_tar = ntuple[:, IDX_XTAR_INIT].astype(np.float32, copy=False)

    l_mag = _compute_lmag(xp_fp, yp_fp, lmag_baseline_cm, lmag_mode, ntuple, lmag_column)
    p_set = np.full(x_fp.shape, np.float32(p0_gev), dtype=np.float32)

    inputs = np.column_stack([x_fp, y_fp, xp_fp, yp_fp, x_tar, p_set, l_mag])

    delta = ntuple[:, IDX_DPP_INIT].astype(np.float32, copy=False)
    y_tar = ntuple[:, IDX_ZTAR_INIT].astype(np.float32, copy=False)
    xptar = ntuple[:, IDX_DPH_INIT].astype(np.float32, copy=False)
    yptar = ntuple[:, IDX_DTH_INIT].astype(np.float32, copy=False)
    targets = np.column_stack([delta, y_tar, xptar, yptar])

    input_features = np.array(
        ["x_fp", "y_fp", "xp_fp", "yp_fp", "x_tar", "p_set", "l_mag"], dtype="<U16"
    )
    target_features = np.array(["delta", "y_tar", "xptar", "yptar"], dtype="<U16")
    return inputs, targets, input_features, target_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align mc-single-arm SHMS npz to SHMS_Calibration_NN format.")
    parser.add_argument("input_npz", help="mc-single-arm npz path (expects key 'ntuple').")
    parser.add_argument(
        "--output",
        help="Output npz path. Defaults to <input>_nn_ready.npz next to input.",
    )
    parser.add_argument(
        "--input-config",
        help="Optional .inp file to infer p_set (GeV/c) and spectrometer angle.",
    )
    parser.add_argument(
        "--p0",
        type=float,
        default=None,
        help="Central momentum (GeV/c). Overrides value parsed from .inp.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep events with stop_id != 0 (default: drop failed events).",
    )
    parser.add_argument(
        "--lmag-mode",
        choices=["zero", "straight", "column"],
        default="zero",
        help="How to build l_mag feature.",
    )
    parser.add_argument(
        "--lmag-baseline-cm",
        type=float,
        default=308.0,
        help="Baseline path length (cm) for --lmag-mode straight.",
    )
    parser.add_argument(
        "--lmag-column",
        type=int,
        default=None,
        help="0-based ntuple column index to use when --lmag-mode column.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    npz_path = Path(args.input_npz).expanduser().resolve()
    data = np.load(npz_path)
    if "ntuple" not in data:
        raise SystemExit(f"'ntuple' key not found in {npz_path}")
    ntuple = data["ntuple"]

    if args.input_config:
        cfg_path = Path(args.input_config).expanduser().resolve()
    else:
        cfg_path = _default_config_from_npz(npz_path)
    p0_gev, theta_deg = (None, None)
    if cfg_path:
        p0_gev, theta_deg = _load_inp_metadata(cfg_path)

    if args.p0 is not None:
        p0_gev = args.p0

    if p0_gev is None:
        raise SystemExit("Central momentum unknown. Provide --p0 or a valid --input-config.")

    inputs, targets, input_features, target_features = _build_output_arrays(
        ntuple,
        p0_gev=p0_gev,
        lmag_mode=args.lmag_mode,
        lmag_baseline_cm=args.lmag_baseline_cm,
        lmag_column=args.lmag_column,
        keep_failed=args.keep_failed,
    )

    out_path = Path(
        args.output
        if args.output
        else npz_path.with_name(f"{npz_path.stem}_nn_ready.npz")
    ).resolve()

    meta: Dict[str, object] = {
        "source_npz": str(npz_path),
        "input_config": str(cfg_path) if cfg_path else None,
        "p_set_gev": p0_gev,
        "theta_deg": theta_deg,
        "lmag_mode": args.lmag_mode,
        "lmag_baseline_cm": args.lmag_baseline_cm,
        "lmag_column": args.lmag_column,
        "kept_failed_events": args.keep_failed,
        "num_events": int(inputs.shape[0]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        inputs=inputs,
        targets=targets,
        input_features=input_features,
        target_features=target_features,
        meta_json=json.dumps(meta, indent=2),
    )

    print(f"Saved NN-aligned file: {out_path}")
    print(f"Events: {inputs.shape[0]}  Inputs shape: {inputs.shape}  Targets shape: {targets.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
