"""On-disk cache for precomputed FastWAM teacher targets (REPA / Channel-1 distillation).

The FastWAM teacher is run **once, offline, over the student's own LIBERO frames**
(see ``tools/cosmos3_precompute_targets.py``) and its frame-0 video-DiT hidden states
are stored here. Because the teacher is applied to the student's frames, cache rows are
keyed by the student dataset's canonical ``(trajectory_id, base_index)`` ordering — there
is **no cross-dataset key matching and no pixel alignment gate**.

On-disk layout (mirrors the project's V-JEPA cache convention)::

    <cache_root>/<dataset_name>/fastwam__<variant>__tap<L>__<mode>/
        meta.json              # JSON dict, see build_meta()
        targets.f16.memmap     # float16 memmap, shape [n_samples, target_dim]
        valid.npy              # bool ndarray [n_samples] (True == usable target)

``target_dim`` and the meaning of each row depend on ``mode``:
  - ``"pooled"``  : per-camera mean-pool of frame-0 hidden -> ``n_cam * teacher_dim``.
  - ``"spatial"`` : full frame-0 grid flattened row-major -> ``h * w * teacher_dim``
                    (reshape to ``[h, w, teacher_dim]`` at load; split cams by width).

A ``steps_hash`` (SHA256 of the ordered ``(str(traj), int(base))`` pairs) is recorded so a
stale cache (dataset membership / ordering changed) hard-fails instead of silently
mis-aligning targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FORMAT_VERSION = 1
TARGETS_FILENAME = "targets.f16.memmap"
VALID_FILENAME = "valid.npy"
META_FILENAME = "meta.json"
TARGET_DTYPE = np.float16

VALID_MODES = ("pooled", "spatial")


def target_dim(mode: str, teacher_dim: int, n_cam: int, grid_hw: tuple[int, int] | None) -> int:
    """Width of one REPA (Channel-1) cache row for the given mode."""
    mode = str(mode).lower()
    if mode == "pooled":
        return int(teacher_dim) * int(n_cam)
    if mode == "spatial":
        if grid_hw is None:
            raise ValueError("`spatial` mode requires grid_hw=(h, w).")
        h, w = grid_hw
        return int(teacher_dim) * int(h) * int(w)
    raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}.")


def future_target_dim(n_future: int, channels: int, grid_hw: tuple[int, int]) -> int:
    """Width of one Channel-2 (future-latent) cache row: n_future * C * h * w (flattened)."""
    h, w = grid_hw
    return int(n_future) * int(channels) * int(h) * int(w)


def cache_subdir_name(variant: str, tap_layer: int, mode: str) -> str:
    """Per-dataset cache subdirectory name. Encodes every key field."""
    safe_variant = "".join(c if c.isalnum() else "_" for c in str(variant)).strip("_")
    return f"fastwam__{safe_variant}__tap{int(tap_layer)}__{str(mode).lower()}"


def steps_hash(all_steps: Iterable[tuple[Any, int]]) -> str:
    """Deterministic SHA256 of the ordered ``(trajectory_id, base_index)`` pairs."""
    hasher = hashlib.sha256()
    for traj_id, base_index in all_steps:
        hasher.update(repr((str(traj_id), int(base_index))).encode("utf-8"))
    return hasher.hexdigest()


def build_meta(
    *,
    variant: str,
    tap_layer: int,
    mode: str,
    n_samples: int,
    teacher_dim: int,
    n_cam: int,
    grid_hw: tuple[int, int] | None,
    dataset_name: str,
    all_steps: Iterable[tuple[Any, int]],
    proprio: bool,
    prompt_template: str,
    target_dim_override: int | None = None,
    n_future: int | None = None,
    channels: int | None = None,
) -> dict:
    """Assemble the ``meta.json`` payload.

    For the Channel-2 future cache (``mode='future'``) pass ``target_dim_override``,
    ``n_future`` and ``channels`` (the REPA ``target_dim`` formula does not apply).
    """
    h, w = (None, None) if grid_hw is None else (int(grid_hw[0]), int(grid_hw[1]))
    tdim = int(target_dim_override) if target_dim_override is not None else int(
        target_dim(mode, teacher_dim, n_cam, grid_hw)
    )
    return {
        "format_version": FORMAT_VERSION,
        "variant": str(variant),
        "tap_layer": int(tap_layer),
        "mode": str(mode).lower(),
        "n_samples": int(n_samples),
        "teacher_dim": int(teacher_dim),
        "n_cam": int(n_cam),
        "grid_h": h,
        "grid_w": w,
        "n_future": None if n_future is None else int(n_future),
        "channels": None if channels is None else int(channels),
        "target_dim": tdim,
        "dtype": "float16",
        "dataset_name": str(dataset_name),
        "targets_file": TARGETS_FILENAME,
        "valid_file": VALID_FILENAME,
        "steps_hash": steps_hash(all_steps),
        "proprio": bool(proprio),
        "prompt_template": str(prompt_template),
    }


def write_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / (META_FILENAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    tmp.replace(cache_dir / META_FILENAME)


def read_meta(cache_dir: Path) -> dict:
    with open(Path(cache_dir) / META_FILENAME) as f:
        return json.load(f)


def resolve_dataset_cache_dir(
    cache_path: str | Path,
    dataset_name: str,
    variant: str,
    tap_layer: int,
    mode: str,
) -> Path:
    """Resolve the concrete cache dir from a (possibly root) ``cache_path``.

    Resolution order:
      1. ``cache_path`` itself if it directly contains ``meta.json``.
      2. ``cache_path/<dataset_name>/<canonical-subdir>``.
      3. Auto-discovery: exactly one ``fastwam__*`` dir with ``meta.json`` under
         ``cache_path/<dataset_name>`` -> use it.
      4. Fall back to the canonical path from (2) (may not exist yet).
    """
    cache_path = Path(cache_path)
    if (cache_path / META_FILENAME).exists():
        return cache_path
    canonical = cache_path / dataset_name / cache_subdir_name(variant, tap_layer, mode)
    if (canonical / META_FILENAME).exists():
        return canonical
    dataset_dir = cache_path / dataset_name
    if dataset_dir.is_dir():
        candidates = [
            d for d in sorted(dataset_dir.glob("fastwam__*")) if (d / META_FILENAME).exists()
        ]
        if len(candidates) == 1:
            return candidates[0]
    return canonical


def validate_meta(meta: dict, *, expected_steps_hash: str, expected_n_samples: int) -> None:
    """Hard-fail on any mismatch between a cache and the live dataset/config."""
    if int(meta.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"FastWAM cache format_version={meta.get('format_version')} != {FORMAT_VERSION}. "
            "Re-run tools/cosmos3_precompute_targets.py."
        )
    if int(meta["n_samples"]) != int(expected_n_samples):
        raise ValueError(
            f"FastWAM cache n_samples={meta['n_samples']} != dataset n_samples={expected_n_samples}. "
            "The dataset changed; re-run the precompute."
        )
    if str(meta["steps_hash"]) != str(expected_steps_hash):
        raise ValueError(
            "FastWAM cache steps_hash mismatch: the (trajectory_id, base_index) ordering "
            "changed between precompute and training. Re-run the precompute."
        )


def open_targets_memmap(cache_dir: Path, mode: str = "r") -> np.memmap:
    """Open the targets memmap read-only (or 'r+'). Shape inferred from meta.json."""
    cache_dir = Path(cache_dir)
    meta = read_meta(cache_dir)
    return np.memmap(
        cache_dir / meta["targets_file"],
        dtype=TARGET_DTYPE,
        mode=mode,
        shape=(int(meta["n_samples"]), int(meta["target_dim"])),
    )


def allocate_targets_memmap(cache_dir: Path, n_samples: int, target_dim_: int) -> np.memmap:
    """Create (zeroed) the targets memmap on disk at full size."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return np.memmap(
        cache_dir / TARGETS_FILENAME,
        dtype=TARGET_DTYPE,
        mode="w+",
        shape=(int(n_samples), int(target_dim_)),
    )
