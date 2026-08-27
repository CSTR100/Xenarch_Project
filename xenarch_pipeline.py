"""
Harness-facing pipeline wrapper.

`run_pipeline(image_path, config) -> List[Dict]` is the interface `eval_harness.py`
scores against. By default it runs the **Mk19 engine from `xenarch_core`** — the
same code path the web app serves — so harness numbers describe the model that
actually ships.

A legacy Mk17 scorer is retained behind `config["engine"] = "mk17"` purely so the
two can be compared on the same data. It is not the product; do not tune it.

    from xenarch_pipeline import run_pipeline
    chips = run_pipeline("scene.png")                      # Mk19 (default)
    chips = run_pipeline("scene.png", {"engine": "mk17"})  # legacy, for comparison

Every returned chip carries:
    chip_id, center_x, center_y, top_x, top_y, source,
    mse/latent/contextual/gradient/edge  (raw and _norm),
    combined, confidence, rank, is_anomaly,
    feature_bbox, fine_bbox, heatmap_peak
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, generic_filter, label

import xenarch_core as core
from xenarch_core import (  # re-exported for callers that imported them from here
    METRIC_KEYS,
    COMBINED_WEIGHTS,
    HAS_TORCH,
    extract_chips,
    load_image_as_array,
    localize_within_chip,
)

# Mk19 defaults, plus the engine switch.
DEFAULT_CONFIG: Dict = {**core.DEFAULT_CONFIG, "engine": "mk19"}

# ── legacy Mk17 weights (comparison engine only) ────────────────────────────
LEGACY_COMBINED_WEIGHTS: Dict[str, float] = {
    "mse": 0.60, "density": 0.30, "contextual": 0.05, "gradient": 0.05, "edge": 0.00,
}
LEGACY_CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "score": 0.50, "contextual": 0.00, "mse": 0.50,
}


# ─────────────────────────────────────────────────────────────────────────────
# MK19 (default)
# ─────────────────────────────────────────────────────────────────────────────

def _run_mk19(image_path: str, cfg: Dict) -> List[Dict]:
    work_dir = tempfile.mkdtemp(prefix="xenarch_pipe_")
    try:
        out = core.analyze_images([image_path], cfg, work_dir=work_dir)
        # Detach from the scratch dir before it is removed: the harness only
        # needs metrics and coordinates, never the chip .npy files.
        chips = []
        for c in out["chips"]:
            c = dict(c)
            c.pop("chip_path", None)
            chips.append(c)
        return chips
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY MK17 SCORER  (retained only to A/B against Mk19 on the same data)
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max. Superseded by robust median/MAD in Mk19 — one extreme chip
    compresses everything else into a narrow band. Kept to reproduce Mk17."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


class LegacyNumpyScorer:
    """Mk17-era hand-crafted scorer. Frozen; do not extend."""

    def __init__(self, reference_chips: List[np.ndarray]):
        if reference_chips:
            stacked = np.stack([c.ravel() for c in reference_chips[:200]])
            self.ref_mean = stacked.mean(axis=0)
            self.ref_std = stacked.std(axis=0) + 1e-8
        else:
            self.ref_mean = None
            self.ref_std = None

    def mse(self, chip: np.ndarray) -> float:
        if self.ref_mean is None:
            return float(chip.std())
        return float(np.mean((chip.ravel() - self.ref_mean) ** 2))

    def density(self, chip: np.ndarray) -> float:
        if self.ref_mean is None:
            return float(np.abs(chip - chip.mean()).mean())
        diff = chip.ravel() - self.ref_mean
        return float(np.sqrt(np.sum((diff / self.ref_std) ** 2)) / chip.size)

    def contextual(self, chip: np.ndarray) -> Tuple[float, Optional[List]]:
        m, s = chip.mean(), chip.std() + 1e-8
        bright_mask = chip > m + 2 * s
        brightness_a = 0.0
        if bright_mask.sum() >= 5:
            brightness_a = min((np.mean(chip[bright_mask]) - m) / (s * 3), 1.0)
        try:
            ls = generic_filter(chip, np.std, size=9)
            texture_a = float((np.abs(ls - ls.mean()) > 2 * (ls.std() + 1e-8)).mean())
        except Exception:
            texture_a = 0.0
        labeled_r, n = label(bright_mask)
        comp_a = 0.0
        for rid in range(1, n + 1):
            reg = labeled_r == rid
            if reg.sum() < 4:
                continue
            ys, xs = np.where(reg)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            c = reg.sum() / ((y2 - y1 + 1) * (x2 - x1 + 1) + 1e-8)
            comp_a = max(comp_a, c)
        score = float(0.35 * brightness_a + 0.35 * texture_a + 0.30 * comp_a)
        return score, self._edge_bbox(chip)

    def _edge_bbox(self, chip: np.ndarray, block: int = 32) -> Optional[List]:
        h, w = chip.shape
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        edge_map = np.sqrt(gx ** 2 + gy ** 2)
        strong = edge_map > np.percentile(edge_map, 85)
        best_score, best_box = -1.0, None
        for y in range(0, h - block + 1, block // 2):
            for x in range(0, w - block + 1, block // 2):
                patch = strong[y:y + block, x:x + block]
                n = patch.sum()
                if n < 5:
                    continue
                patch_f = patch.astype(np.float32)
                total = float(n) + 1e-8
                row_align = patch_f.sum(axis=1).max() / total
                col_align = patch_f.sum(axis=0).max() / total
                angles = np.arctan2(gy[y:y + block, x:x + block][patch],
                                    gx[y:y + block, x:x + block][patch])
                R = float(np.abs(np.mean(np.exp(2j * angles))))
                reg_score = float(max(row_align, col_align)) * float(patch_f.mean()) * R
                if reg_score > best_score:
                    best_score = reg_score
                    best_box = [y / h, x / w, (y + block) / h, (x + block) / w]
        return best_box

    def gradient(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        gm = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.abs(gm - gaussian_filter(gm, sigma=4)).mean())

    def edge(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        es = np.sqrt(gx ** 2 + gy ** 2)
        strong = es > np.percentile(es, 90)
        if strong.sum() < 10:
            return 0.0
        ra = strong.sum(axis=1).max() / (strong.sum() + 1e-8)
        ca = strong.sum(axis=0).max() / (strong.sum() + 1e-8)
        return float(max(ra, ca))


def _run_mk17(image_path: str, cfg: Dict,
              reference_chips: Optional[List[np.ndarray]]) -> List[Dict]:
    chip_size = int(cfg.get("chip_size", 256))
    combined_w = cfg.get("combined_weights", LEGACY_COMBINED_WEIGHTS)
    confidence_w = cfg.get("confidence_weights", LEGACY_CONFIDENCE_WEIGHTS)

    work_dir = tempfile.mkdtemp(prefix="xenarch_mk17_")
    try:
        # Mk17 tiled without overlap.
        chips_meta = core.extract_chips(image_path, work_dir, chip_size=chip_size,
                                        overlap=0.0, max_chips=600)
        if not chips_meta:
            return []
        chips_arr = [np.load(c["chip_path"]) for c in chips_meta]

        scorer = LegacyNumpyScorer(reference_chips if reference_chips is not None
                                   else chips_arr)
        raw: List[Dict] = []
        for chip in chips_arr:
            ctx_score, bbox = scorer.contextual(chip)
            raw.append({
                "mse":          scorer.mse(chip),
                "density":      scorer.density(chip),
                "contextual":   ctx_score,
                "gradient":     scorer.gradient(chip),
                "edge":         scorer.edge(chip),
                "feature_bbox": bbox,
            })

        keys = ["mse", "density", "contextual", "gradient", "edge"]
        norms = {k: _legacy_normalize(np.array([r[k] for r in raw])) for k in keys}

        out: List[Dict] = []
        for i, r in enumerate(raw):
            r2 = dict(r)
            for k in keys:
                r2[f"{k}_norm"] = float(norms[k][i])
            # Mk17's "density" plays the role Mk19 calls "latent"; alias it so
            # both engines hand the harness one schema.
            r2["latent"] = r2["density"]
            r2["latent_norm"] = r2["density_norm"]
            r2["combined"] = float(sum(combined_w.get(k, 0.0) * norms[k][i] for k in keys))
            out.append(r2)

        sn = _legacy_normalize(np.array([r["combined"] for r in out]))
        conf = np.clip(
            confidence_w.get("score", 0.5) * sn
            + confidence_w.get("contextual", 0.0) * np.array([r["contextual_norm"] for r in out])
            + confidence_w.get("mse", 0.5) * np.array([r["mse_norm"] for r in out]),
            0.0, 1.0)
        for i, r in enumerate(out):
            r["confidence"] = float(conf[i])

        results = [{**meta, **score} for meta, score in zip(chips_meta, out)]
        results.sort(key=lambda x: x["confidence"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
            r["is_anomaly"] = None

        # stage-2 localization on the top-N, same as Mk19
        top_n = int(cfg.get("localize_top_n", 5))
        chip_by_id = {m["chip_id"]: arr for m, arr in zip(chips_meta, chips_arr)}
        for r in results:
            r["fine_bbox"] = None
            r["heatmap_peak"] = None
        for r in results[:top_n]:
            r.update(core.localize_within_chip(chip_by_id[r["chip_id"]],
                                               r["top_x"], r["top_y"]))
        for r in results:
            r.pop("chip_path", None)
        return results
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_reference_corpus(manifest_path: str) -> List[np.ndarray]:
    """Load chip arrays from a manifest.json listing {"chips": [{"chip_path": ...}]}."""
    import json
    manifest = json.loads(Path(manifest_path).read_text())
    return [np.load(e["chip_path"]) for e in manifest.get("chips", [])
            if Path(e.get("chip_path", "")).exists()]


def run_pipeline(image_path: str,
                 config: Optional[Dict] = None,
                 reference_chips: Optional[List[np.ndarray]] = None,
                 bg_chip_paths: Optional[List[str]] = None) -> List[Dict]:
    """Run anomaly detection on one image and return all chips, ranked.

    config["engine"]:
        "mk19" (default) — the shipping model. Trains on the natural-geology
                           baseline in `training data/` (or config["training_dir"]),
                           falling back to self-supervised trimmed training.
        "mk17"           — frozen legacy scorer, for comparison only.

    reference_chips / bg_chip_paths apply to the legacy engine only; Mk19 takes
    its baseline from the training directory.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    engine = str(cfg.get("engine", "mk19")).lower()

    if engine == "mk17":
        return _run_mk17(image_path, cfg, reference_chips)
    if engine != "mk19":
        raise ValueError(f"unknown engine {engine!r} — expected 'mk19' or 'mk17'")
    return _run_mk19(image_path, cfg)
