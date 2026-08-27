"""
Build the evaluation fixtures the harness expects.

`eval_harness.py` needs labeled regions: imagery with known artificial features
at known coordinates, plus known-hard natural features that a detector must NOT
flag. Real spacecraft imagery does not come with that labeling, so this script
manufactures it — either by injecting synthetic anomalies into your own natural
source imagery, or, with --demo, into procedurally generated terrain.

    # Self-contained demo: generate terrain, inject, write eval/ + training data/
    python inject_anomalies.py --demo

    # Use your own natural-geology imagery as the substrate
    python inject_anomalies.py --source path/to/natural/images

    # Reproducible variations
    python inject_anomalies.py --demo --seed 7

Outputs:
    eval/regions.json                     region list, splits, anomaly types
    eval/<region_id>/scene.png            the injected scene
    eval/<region_id>/ground_truth.json    zones: target / false_positive / background
    training data/<name>.png              (demo only) clean natural baseline

IMPORTANT — what these fixtures are and are not:
  The injected features are SYNTHETIC. They measure whether the detector
  separates hard-edged artificial geometry from natural terrain, which is a
  necessary property, not a sufficient one. Good separation here does not
  establish performance on real orbital imagery of real hardware. Treat these
  numbers as a regression guard while tuning, and validate on real scenes
  separately.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
IMG_SIZE = 1024
CHIP_SIZE = 256
CHIP_STRIDE = 128     # chip_size 256 at 50% overlap -> chip centers on this grid
SRC_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC NATURAL TERRAIN
# ─────────────────────────────────────────────────────────────────────────────

def _value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Smooth value noise at one octave, bilinearly upsampled from a coarse grid."""
    coarse = rng.random((cells + 1, cells + 1))
    img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0


def fractal_terrain(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Sum of octaves — the 1/f statistics that make natural surfaces look natural."""
    out = np.zeros((size, size), dtype=np.float32)
    amp, total = 1.0, 0.0
    for cells in (2, 4, 8, 16, 32, 64, 128):
        out += amp * _value_noise(rng, size, cells)
        total += amp
        amp *= 0.55
    out /= total
    return out


def add_crater(arr: np.ndarray, cx: int, cy: int, radius: float,
               rng: np.random.Generator, bright_rim: bool = False) -> None:
    """Bowl with a raised rim. The classic natural false positive: high contrast,
    strong edges, radially symmetric."""
    h, w = arr.shape
    y0, y1 = max(0, int(cy - radius * 2)), min(h, int(cy + radius * 2))
    x0, x1 = max(0, int(cx - radius * 2)), min(w, int(cx + radius * 2))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(yy - cy, xx - cx) / radius

    bowl = -0.18 * np.exp(-(r ** 2) * 1.6)          # depression
    rim = 0.22 * np.exp(-((r - 1.0) ** 2) / 0.05)   # raised rim
    if bright_rim:
        rim *= 1.9
    # illumination asymmetry: one side of the bowl shadowed
    shade = -0.10 * np.clip(np.cos(np.arctan2(yy - cy, xx - cx)), 0, 1) * np.exp(-(r ** 2) * 1.2)

    arr[y0:y1, x0:x1] += (bowl + rim + shade).astype(np.float32)


def add_boulder_field(arr: np.ndarray, cx: int, cy: int, extent: int,
                      rng: np.random.Generator, n: int = 40) -> None:
    """Scattered bright specks with dark shadows. Natural, but bright and compact —
    exactly the texture that generates false alarms."""
    h, w = arr.shape
    for _ in range(n):
        bx = int(np.clip(cx + rng.normal(0, extent / 3), 2, w - 3))
        by = int(np.clip(cy + rng.normal(0, extent / 3), 2, h - 3))
        rad = int(rng.integers(1, 4))
        arr[by - rad:by + rad + 1, bx - rad:bx + rad + 1] += 0.30
        arr[by:by + rad + 2, bx - rad - 1:bx] -= 0.18   # shadow


def natural_scene(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Terrain + a scattering of craters. No artificial content."""
    arr = fractal_terrain(rng, size)
    for _ in range(int(rng.integers(6, 12))):
        add_crater(arr, int(rng.integers(60, size - 60)), int(rng.integers(60, size - 60)),
                   float(rng.uniform(14, 45)), rng)
    return np.clip(arr, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# INJECTED ANOMALIES  (the "technosignature" targets)
# ─────────────────────────────────────────────────────────────────────────────

def inject_rectilinear(arr: np.ndarray, cx: int, cy: int,
                       rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """Hard-edged rectangular platform with a cast shadow. Straight edges,
    right angles, uniform interior — none of which erosion produces."""
    hw, hh = int(rng.integers(14, 26)), int(rng.integers(10, 20))
    arr[cy - hh:cy + hh, cx - hw:cx + hw] = 0.86
    arr[cy - hh + 3:cy + hh - 3, cx - hw + 3:cx + hw - 3] = 0.74   # inner panel
    arr[cy + hh:cy + hh + 7, cx - hw + 4:cx + hw + 4] = 0.08       # shadow
    return (cx - hw, cy - hh, cx + hw + 4, cy + hh + 7)


def inject_linear(arr: np.ndarray, cx: int, cy: int,
                  rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """A long straight track at an arbitrary angle — the case axis-aligned edge
    detectors miss and the FFT angular-concentration term is meant to catch."""
    length = int(rng.integers(120, 200))
    angle = float(rng.uniform(0, np.pi))
    dx, dy = np.cos(angle), np.sin(angle)
    h, w = arr.shape
    for t in np.linspace(-length / 2, length / 2, length * 3):
        x, y = int(cx + dx * t), int(cy + dy * t)
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            arr[y - 1:y + 2, x - 1:x + 2] = 0.82
    ex, ey = int(abs(dx) * length / 2) + 2, int(abs(dy) * length / 2) + 2
    return (cx - ex, cy - ey, cx + ex, cy + ey)


def inject_compact_object(arr: np.ndarray, cx: int, cy: int,
                          rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """Small, very bright, compact, with a hard shadow — a lander-like signature.
    Deliberately the smallest target: this is what patch-max error exists for."""
    r = int(rng.integers(5, 9))
    arr[cy - r:cy + r, cx - r:cx + r] = 0.94
    arr[cy - r - 2:cy - r, cx - 2:cx + 2] = 0.88      # mast / antenna
    arr[cy + r:cy + r + 9, cx - r + 3:cx + r + 3] = 0.05   # long shadow
    return (cx - r, cy - r - 2, cx + r + 3, cy + r + 9)


def inject_grid(arr: np.ndarray, cx: int, cy: int,
                rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """Repeated elements on a regular pitch. Periodicity at a fixed spacing is a
    spectral signature natural terrain does not produce."""
    pitch = int(rng.integers(11, 16))
    n = 4
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = cx + i * pitch, cy + j * pitch
            arr[y - 2:y + 3, x - 2:x + 3] = 0.88
    ext = n * pitch + 2
    return (cx - ext, cy - ext, cx + ext, cy + ext)


INJECTORS = {
    "rectilinear_structure": inject_rectilinear,
    "linear_feature":        inject_linear,
    "compact_bright_object": inject_compact_object,
    "regular_grid":          inject_grid,
}


# ─────────────────────────────────────────────────────────────────────────────
# REGION ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def _snap_to_chip_center(v: int) -> int:
    """Put a feature on the chip-center lattice so at least one chip center
    lands inside its zone regardless of how tiling falls."""
    return int(round((v - CHIP_STRIDE) / CHIP_STRIDE)) * CHIP_STRIDE + CHIP_STRIDE


def _containment_zone(feat_bbox: Tuple[int, int, int, int], label: str,
                      chip_size: int, size: int) -> Dict:
    """Zone covering exactly the chip centers whose chip FULLY CONTAINS the feature.

    The harness labels a chip by whether its center falls inside a zone. A chip
    of width `chip_size` centred at c spans [c - chip_size/2, c + chip_size/2],
    so it fully contains a feature spanning [f1, f2] when

        c >= f2 - chip_size/2   and   c <= f1 + chip_size/2

    Deriving the zone this way — rather than from a fixed half-width — matters:
    too wide and chips holding only a clipped corner of the feature get labeled
    `target`, which drags target confidence down and understates separation;
    too narrow and chips that genuinely contain the feature are scored as
    background. Either way the harness measures the fixture, not the model.
    """
    x1, y1, x2, y2 = feat_bbox
    half = chip_size // 2
    zx1, zx2 = x2 - half, x1 + half
    zy1, zy2 = y2 - half, y1 + half
    # A feature larger than a chip inverts the interval; fall back to the chip
    # centred on it, which is the best any tiling can do.
    if zx1 > zx2:
        zx1 = zx2 = (x1 + x2) // 2
    if zy1 > zy2:
        zy1 = zy2 = (y1 + y2) // 2
    return {
        "label": label,
        "bbox": [max(0, int(zx1)), max(0, int(zy1)),
                 min(size - 1, int(zx2)), min(size - 1, int(zy2))],
    }


def build_region(region_id: str,
                 anomaly_type: str,
                 rng: np.random.Generator,
                 substrate: Optional[np.ndarray] = None,
                 size: int = IMG_SIZE,
                 chip_size: int = CHIP_SIZE) -> Tuple[np.ndarray, Dict]:
    """Return (image, ground_truth). One injected target, two natural
    false-positive features, background everywhere else."""
    arr = natural_scene(rng, size) if substrate is None else substrate.copy()

    margin = 260
    sep = chip_size + 64        # keep features from sharing chips
    zones: List[Dict] = []
    taken: List[Tuple[int, int]] = []

    def pick_spot() -> Tuple[int, int]:
        cx = cy = size // 2
        for _ in range(400):
            cx = _snap_to_chip_center(int(rng.integers(margin, size - margin)))
            cy = _snap_to_chip_center(int(rng.integers(margin, size - margin)))
            if all(abs(cx - px) > sep or abs(cy - py) > sep for px, py in taken):
                break
        taken.append((cx, cy))
        return cx, cy

    # ── the target ──────────────────────────────────────────────────────────
    tx, ty = pick_spot()
    tbox = INJECTORS[anomaly_type](arr, tx, ty, rng)
    zones.append(_containment_zone(tbox, "target", chip_size, size))

    # ── natural features that a naive detector mistakes for targets ─────────
    fx, fy = pick_spot()
    crater_r = float(rng.uniform(30, 46))
    add_crater(arr, fx, fy, crater_r, rng, bright_rim=True)
    ce = int(crater_r * 2)
    zones.append(_containment_zone((fx - ce, fy - ce, fx + ce, fy + ce),
                                   "false_positive", chip_size, size))

    bx, by = pick_spot()
    add_boulder_field(arr, bx, by, 70, rng)
    zones.append(_containment_zone((bx - 70, by - 70, bx + 70, by + 70),
                                   "false_positive", chip_size, size))

    # background last: first match wins, so the labeled zones above take priority
    zones.append({"label": "background", "bbox": [0, 0, size - 1, size - 1]})

    arr = np.clip(arr, 0.0, 1.0)
    gt = {
        "region_id":      region_id,
        "image_path":     f"eval/{region_id}/scene.png",
        "anomaly_type":   anomaly_type,
        "synthetic":      True,
        "low_confidence": False,
        "target_feature_bbox": [int(v) for v in tbox],
        "zones":          zones,
    }
    return arr, gt


REGION_PLAN = [
    ("region_01_rect_tuning",   "rectilinear_structure", "tuning"),
    ("region_02_compact_tuning", "compact_bright_object", "tuning"),
    ("region_03_linear_tuning",  "linear_feature",        "tuning"),
    ("region_04_rect_held",     "rectilinear_structure", "held-out"),
    ("region_05_compact_held",  "compact_bright_object", "held-out"),
    ("region_06_grid_held",     "regular_grid",          "held-out"),
]


def load_substrates(source_dir: Path, size: int) -> List[np.ndarray]:
    """Load real natural imagery to inject into, center-cropped/resized to size."""
    paths = sorted(p for p in source_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SRC_EXTS)
    out = []
    for p in paths:
        try:
            if p.suffix.lower() == ".npy":
                a = np.load(p).astype(np.float32)
                if a.ndim == 3:
                    a = a.mean(axis=-1)
            else:
                a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            lo, hi = np.percentile(a, [1, 99])
            a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
            img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
            out.append(np.asarray(img.resize((size, size), Image.BICUBIC),
                                  dtype=np.float32) / 255.0)
        except Exception as e:
            print(f"  skipping {p.name}: {e}")
    return out


def save_gray(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate labeled evaluation fixtures for eval_harness.py")
    ap.add_argument("--demo", action="store_true",
                    help="Generate synthetic terrain instead of using real imagery, "
                         "and populate 'training data/' with clean natural scenes.")
    ap.add_argument("--source", type=str, default=None,
                    help="Directory of natural-geology imagery to inject into.")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    ap.add_argument("--size", type=int, default=IMG_SIZE, help="Output image size.")
    ap.add_argument("--chip-size", type=int, default=CHIP_SIZE,
                    help="Chip size the harness will use; sets zone half-width.")
    ap.add_argument("--n-train", type=int, default=4,
                    help="Demo mode: how many clean baseline scenes to write.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing eval/ directory.")
    args = ap.parse_args()

    if not args.demo and not args.source:
        ap.error("pass --demo, or --source DIR with your own natural imagery")

    rng = np.random.default_rng(args.seed)
    eval_dir = BASE / "eval"
    if eval_dir.exists() and any(eval_dir.iterdir()):
        if not args.force:
            ap.error(f"{eval_dir} already exists and is not empty — pass --force to overwrite")
        shutil.rmtree(eval_dir)

    substrates: List[np.ndarray] = []
    if args.source:
        src = Path(args.source)
        if not src.is_dir():
            ap.error(f"--source directory not found: {src}")
        substrates = load_substrates(src, args.size)
        if not substrates:
            ap.error(f"no usable imagery in {src}")
        print(f"Loaded {len(substrates)} substrate image(s) from {src}")

    # ── demo: a clean natural baseline for the VAE to train on ──────────────
    if args.demo:
        train_dir = BASE / "training data"
        train_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_train):
            save_gray(natural_scene(rng, args.size),
                      train_dir / f"synthetic_natural_{i:02d}.png")
        print(f"Wrote {args.n_train} clean baseline scene(s) → {train_dir}")

    # ── regions ─────────────────────────────────────────────────────────────
    regions_meta = []
    for i, (rid, atype, split) in enumerate(REGION_PLAN):
        sub = substrates[i % len(substrates)] if substrates else None
        arr, gt = build_region(rid, atype, rng, substrate=sub, size=args.size,
                               chip_size=args.chip_size)
        save_gray(arr, eval_dir / rid / "scene.png")
        (eval_dir / rid / "ground_truth.json").write_text(json.dumps(gt, indent=2))
        regions_meta.append({
            "region_id":    rid,
            "split":        split,
            "anomaly_type": atype,
            "chip_size":    args.chip_size,
            "synthetic":    True,
        })
        tz = next(z for z in gt["zones"] if z["label"] == "target")
        print(f"  {rid:<28} {split:<9} {atype:<22} target@{tz['bbox']}")

    (eval_dir / "regions.json").write_text(
        json.dumps({"regions": regions_meta}, indent=2))

    print(f"\nWrote {len(regions_meta)} regions → {eval_dir}")
    print("These targets are SYNTHETIC — a regression guard, not evidence of "
          "real-imagery performance.")
    print("Next:  python eval_harness.py --iteration 1")


if __name__ == "__main__":
    main()
