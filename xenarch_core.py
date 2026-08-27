"""
Xenarch Mk19 core — the anomaly-detection engine, free of Flask and of any
web-server concerns.

This module is the single source of truth for the Mk19 algorithm. Both
consumers import it rather than reimplementing it:

    xenarch_mk19_script.py   Flask app + embedded frontend  → analyze_images()
    xenarch_pipeline.py      harness-facing run_pipeline()  → analyze_images()

Keeping one implementation here is what lets `eval_harness.py` measure the
model that actually ships. Previously the harness scored an older Mk17-era
copy of the logic, so its numbers described a model nobody ran.

Pipeline (see README for the reasoning behind each step):
  1. chip extraction, 50% overlap
  2. VAE trained on the natural-geology baseline, with trimmed robust
     training + augmentation so anomalies are never learned as normal
  3. five metrics: patch-max reconstruction error, latent distance,
     two-sided contextual, gradient irregularity, edge regularity
  4. robust (median/MAD) normalization against the baseline distribution
  5. confidence = sigmoid of the combined robust z, centered at z=2
  6. optional stage-2 fine localization within the top-ranked chips
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label, uniform_filter

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CHIP_SIZE = 256

METRIC_KEYS = ["mse", "latent", "contextual", "gradient", "edge"]

# Combined-score weights. Edge regularity (straight lines at any orientation)
# is the strongest geology-vs-technology discriminator, so it carries real
# weight. Patch-max reconstruction error remains the primary VAE signal.
COMBINED_WEIGHTS = {
    "mse":        0.30,   # patch-wise MAX reconstruction error
    "latent":     0.15,   # robust latent-space distance
    "contextual": 0.20,   # compact bright OR dark feature + texture outlier
    "gradient":   0.10,   # local gradient irregularity
    "edge":       0.25,   # orientation-invariant edge regularity
}

# The VAE trains on imagery found here (the curated "all natural geology" set)
# and scenes are scored against that baseline. Override with the
# XENARCH_TRAINING_DIR env var or per-job with config["training_dir"].
TRAINING_DIR = Path(os.environ.get(
    "XENARCH_TRAINING_DIR",
    str(Path(__file__).resolve().parent / "training data")))
TRAIN_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}

# Defaults for every tunable. Callers pass overrides in a plain dict.
DEFAULT_CONFIG: Dict = {
    "chip_size":       CHIP_SIZE,
    "overlap":         0.5,
    "percentile":      92,
    "epochs":          20,
    "latent_dim":      56,
    "batch_size":      4,
    "lr":              0.0005,
    "warmup_epochs":   3,
    "trim_frac":       0.08,
    "max_train_chips": 800,
    "max_chips":       500,
    "localize_top_n":  5,
    "training_dir":    None,
    "seed":            0,      # None = nondeterministic
}


def list_training_images(train_dir) -> List[Path]:
    try:
        d = Path(train_dir)
        if not d.is_dir():
            return []
        return sorted(p for p in d.rglob("*")
                      if p.is_file() and p.suffix.lower() in TRAIN_EXTS)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SMALL NUMERIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-scores. Immune to a single extreme outlier, unlike min-max."""
    v = np.asarray(values, dtype=np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    if mad < 1e-12:
        mad = v.std() + 1e-12
    return (v - med) / mad


def robust_norm(values: np.ndarray) -> np.ndarray:
    """Squash robust z-scores through a sigmoid into (0,1)."""
    return 1.0 / (1.0 + np.exp(-robust_z(values) / 2.0))


def local_std(a: np.ndarray, size: int = 9) -> np.ndarray:
    """Fast closed-form local std (replaces the very slow generic_filter(np.std))."""
    m = uniform_filter(a, size=size, mode="reflect")
    m2 = uniform_filter(a * a, size=size, mode="reflect")
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_image_as_array(path: str) -> np.ndarray:
    """Return a float32 grayscale array in [0,1], 1st-99th percentile stretched."""
    if str(path).lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-8:
            return np.clip(arr, 0.0, 1.0)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
            return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)
        except Exception:
            pass
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHIP EXTRACTION  (overlapped, so features aren't split at seams)
# ─────────────────────────────────────────────────────────────────────────────

def extract_chips(image_path: str, output_dir: str,
                  chip_size: int = CHIP_SIZE,
                  overlap: float = 0.5,
                  max_chips: int = 500) -> List[Dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    arr = load_image_as_array(image_path)
    h, w = arr.shape
    stride = max(int(round(chip_size * (1.0 - overlap))), 16)
    chips: List[Dict] = []
    chip_id = 0
    source_stem = Path(image_path).stem

    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            if chip_id >= max_chips:
                break
            chip = arr[y:y + chip_size, x:x + chip_size]
            if chip.std() < 0.005:          # drop only truly empty chips
                continue
            chip_path = output_path / f"{source_stem}_chip_{chip_id:04d}.npy"
            np.save(str(chip_path), chip.astype(np.float32))
            chips.append({
                "chip_id":   chip_id,
                "chip_path": str(chip_path),
                "center_x":  x + chip_size // 2,
                "center_y":  y + chip_size // 2,
                "top_x":     x,
                "top_y":     y,
                "source":    Path(image_path).name,
            })
            chip_id += 1
        if chip_id >= max_chips:
            break

    return chips


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NUMPY MULTI-METRIC SCORER  (also the fallback when torch is absent)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyAnomalyScorer:
    """Hand-crafted metrics. When torch is unavailable this is the whole scorer;
    when torch is available it supplies the contextual/gradient/edge metrics."""

    def __init__(self, reference_chips: Optional[List[np.ndarray]] = None):
        self.f_med = None
        self.f_mad = None
        if reference_chips:
            feats = np.stack([self._features(c) for c in reference_chips[:400]])
            self.f_med = np.median(feats, axis=0)
            self.f_mad = np.median(np.abs(feats - self.f_med), axis=0) * 1.4826 + 1e-8

    @staticmethod
    def _features(chip: np.ndarray) -> np.ndarray:
        """Simple statistical fingerprint of a chip, for the fallback latent score."""
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        gm = np.sqrt(gx ** 2 + gy ** 2)
        ls = local_std(chip, 9)
        return np.array([
            chip.mean(), chip.std(),
            gm.mean(), gm.std(),
            ls.mean(), ls.std(),
            np.percentile(chip, 99) - np.percentile(chip, 1),
        ], dtype=np.float64)

    def latent_score(self, chip: np.ndarray) -> float:
        """Fallback 'latent' distance: robust z-distance of the chip's statistical
        fingerprint from the population. (The torch path replaces this with the
        VAE's latent Mahalanobis distance.)"""
        if self.f_med is None:
            return float(chip.std())
        z = (self._features(chip) - self.f_med) / self.f_mad
        return float(np.sqrt(np.mean(z ** 2)))

    def mse_score(self, chip: np.ndarray, patch: int = 32) -> float:
        """Fallback reconstruction-style score: PATCH-WISE MAX deviation from a
        smooth background model, so a small artifact isn't averaged away."""
        smooth = gaussian_filter(chip, sigma=8)
        resid = (chip - smooth) ** 2
        pooled = uniform_filter(resid, size=patch, mode="reflect")
        return float(pooled.max())

    def contextual_score(self, chip: np.ndarray) -> Dict:
        """Compact locally-deviant region, BRIGHT or DARK (shadowed hardware),
        plus texture-outlier fraction."""
        mean_b = chip.mean()
        std_b = chip.std() + 1e-8

        ls = local_std(chip, 9)
        tex_mean = ls.mean()
        tex_std = ls.std() + 1e-8
        texture_a = float((np.abs(ls - tex_mean) > 2 * tex_std).mean())

        best_score, best_bbox = 0.0, None
        for mask in (chip > mean_b + 2 * std_b,      # bright anomaly
                     chip < mean_b - 2 * std_b):     # dark anomaly / shadow
            if mask.sum() < 5:
                continue
            dev = abs(float(chip[mask].mean()) - mean_b)
            intensity_a = min(dev / (std_b * 3), 1.0)

            labeled, n_regions = label(mask)
            comp_a, bbox = 0.0, None
            for rid in range(1, n_regions + 1):
                region = labeled == rid
                size = int(region.sum())
                if size < 4:
                    continue
                ys, xs = np.where(region)
                y1, y2 = int(ys.min()), int(ys.max())
                x1, x2 = int(xs.min()), int(xs.max())
                compactness = size / ((y2 - y1 + 1) * (x2 - x1 + 1) + 1e-8)
                if compactness > comp_a:
                    comp_a = compactness
                    bbox = [y1 / chip.shape[0], x1 / chip.shape[1],
                            y2 / chip.shape[0], x2 / chip.shape[1]]
            score = 0.35 * intensity_a + 0.35 * texture_a + 0.30 * comp_a
            if score > best_score:
                best_score, best_bbox = score, bbox

        return {"score": float(best_score), "bbox": best_bbox}

    def gradient_score(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.abs(grad_mag - gaussian_filter(grad_mag, sigma=4)).mean())

    def edge_regularity_score(self, chip: np.ndarray) -> float:
        """Orientation-invariant straight-edge detector.
        (a) alignment of strong edges along rows/cols (axis-aligned lines) and
        (b) FFT angular-spectrum concentration: natural terrain has an isotropic
            spectrum; straight edges at ANY angle concentrate energy in a narrow
            angular band."""
        h, w = chip.shape
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        edge_str = np.sqrt(gx ** 2 + gy ** 2)
        strong = edge_str > np.percentile(edge_str, 90)
        line_a = 0.0
        if strong.sum() >= 10:
            row_align = strong.sum(axis=1).max() / (strong.sum() + 1e-8)
            col_align = strong.sum(axis=0).max() / (strong.sum() + 1e-8)
            line_a = float(max(row_align, col_align))

        # FFT angular concentration
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        spec = np.abs(np.fft.fftshift(np.fft.fft2((chip - chip.mean()) * win)))
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - cy, xx - cx)
        theta = np.mod(np.arctan2(yy - cy, xx - cx), np.pi)
        sel = (r > 4) & (r < min(h, w) // 2)
        nbins = 36
        hist, _ = np.histogram(theta[sel], bins=nbins, range=(0, np.pi),
                               weights=spec[sel])
        conc = hist.max() / (hist.sum() + 1e-9)
        # uniform spectrum -> conc = 1/nbins; strong linear feature -> conc >> 1/nbins
        fft_a = float(np.clip((conc - 1.0 / nbins) / (0.25 - 1.0 / nbins), 0.0, 1.0))

        return float(0.5 * line_a + 0.5 * fft_a)

    def score(self, chip: np.ndarray) -> Dict:
        ctx = self.contextual_score(chip)
        return {
            "mse":          self.mse_score(chip),
            "latent":       self.latent_score(chip),
            "contextual":   ctx["score"],
            "gradient":     self.gradient_score(chip),
            "edge":         self.edge_regularity_score(chip),
            "feature_bbox": ctx["bbox"],
        }

    @staticmethod
    def fit_norm_stats(score_list: List[Dict]) -> Dict[str, tuple]:
        """Fit robust (median/MAD) normalization statistics — typically on the
        TRAINING baseline, so scene chips are measured against 'natural'."""
        stats = {}
        for k in METRIC_KEYS:
            v = np.array([s[k] for s in score_list], dtype=np.float64)
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)) * 1.4826)
            if mad < 1e-12:
                mad = float(v.std()) + 1e-12
            stats[k] = (med, mad)
        return stats

    @staticmethod
    def apply_norm(score_list: List[Dict], stats: Dict[str, tuple]) -> List[Dict]:
        """Robust z + sigmoid normalization with externally supplied statistics;
        a single extreme chip can't compress the rest the way min-max did."""
        result = []
        for s in score_list:
            s_out = dict(s)
            combined = 0.0
            for k in METRIC_KEYS:
                med, mad = stats[k]
                n = float(1.0 / (1.0 + np.exp(-((s[k] - med) / mad) / 2.0)))
                s_out[f"{k}_norm"] = n
                combined += COMBINED_WEIGHTS[k] * n
            s_out["combined"] = float(combined)
            result.append(s_out)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TORCH VAE
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:

    class StableConvolutionalVAE(nn.Module):
        def __init__(self, latent_dim=56, input_size=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.input_size = input_size
            final_size = input_size // 16
            self.final_size = final_size
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),   nn.BatchNorm2d(32, eps=1e-3),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  nn.BatchNorm2d(64, eps=1e-3),  nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256, eps=1e-3), nn.ReLU(),
            )
            self.fc_mu = nn.Linear(256 * final_size * final_size, latent_dim)
            self.fc_logvar = nn.Linear(256 * final_size * final_size, latent_dim)
            nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
            nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
            self.decoder_input = nn.Linear(latent_dim, 256 * final_size * final_size)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = torch.flatten(self.encoder_conv(x), 1)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            return mu, torch.clamp(logvar, -10, 10)

        @staticmethod
        def reparameterize(mu, logvar):
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        def decode(self, z):
            h = self.decoder_input(z).view(-1, 256, self.final_size, self.final_size)
            return self.decoder_conv(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    class TorchDataset(Dataset):
        """augment=True applies random flips / 90-degree rotations so the VAE
        learns geology statistics rather than memorizing individual chips."""

        def __init__(self, chip_paths, augment=False):
            self.paths = chip_paths
            self.augment = augment

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            arr = np.load(self.paths[idx]).astype(np.float32)
            if self.augment:
                k = np.random.randint(4)
                if k:
                    arr = np.rot90(arr, k)
                if np.random.rand() < 0.5:
                    arr = np.fliplr(arr)
                arr = np.ascontiguousarray(arr)
            return torch.from_numpy(arr[np.newaxis]), self.paths[idx]

    def stable_vae_loss(recon, x, mu, logvar, beta=0.01, kl_weight=1.0):
        # beta 0.01 (not 0.001): more regularization discourages memorization
        # of rare (anomalous) chips.
        mse = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + beta * kl_weight * kld, mse, kld

    def per_chip_recon_mse(model, chip_paths, device, batch_size) -> np.ndarray:
        """Deterministic (mu-path) whole-chip MSE, used to select training inliers."""
        model.eval()
        out = []
        loader = DataLoader(TorchDataset(chip_paths, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                recon = model.decode(model.encode(imgs)[0])
                out.extend(torch.mean((imgs - recon) ** 2, dim=[1, 2, 3]).cpu().numpy().tolist())
        model.train()
        return np.array(out)

    def patchwise_max_error(imgs, recon, patch=32):
        """Local mean of squared error, then MAX over patches. A small artifact
        dominates its chip's score instead of being averaged away."""
        err = (imgs - recon) ** 2
        pooled = F.avg_pool2d(err, kernel_size=patch, stride=max(patch // 2, 1))
        return pooled.flatten(1).max(dim=1).values


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONFIDENCE  (calibrated: low when nothing is truly anomalous)
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(scored_chips: List[Dict],
                       ref_combined: Optional[List[float]] = None) -> List[Dict]:
    """Sigmoid of the robust z of the combined score, centered at z=2.
    A chip 2 robust sigmas above the reference median -> 0.5; 4 sigmas -> ~0.88.
    When a training baseline exists, z is measured against the BASELINE
    distribution ('how far from natural geology?'); otherwise against the scene
    itself. Either way the top chip is NOT automatically 'confident' — if the
    scene is all natural, everything scores low."""
    comb = np.array([c["combined"] for c in scored_chips], dtype=np.float64)
    base = (np.asarray(ref_combined, dtype=np.float64)
            if ref_combined is not None else comb)
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826
    if mad < 1e-12:
        mad = base.std() + 1e-12
    conf = 1.0 / (1.0 + np.exp(-((comb - med) / mad - 2.0)))
    out = []
    for i, c in enumerate(scored_chips):
        c2 = dict(c)
        c2["confidence"] = float(np.clip(conf[i], 0.0, 1.0))
        out.append(c2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  STAGE-2 FINE LOCALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def localize_within_chip(chip: np.ndarray,
                         chip_top_x: int,
                         chip_top_y: int) -> Dict:
    """Dense sliding-window heatmap within one chip, scoring every sub-patch on:
      - MSE vs the chip background mean   (brightness anomalies)
      - local texture deviation           (texture outliers)
      - directional edge regularity       (straight edges; penalises crater rims)

    Returns fine_bbox [x1,y1,x2,y2] in IMAGE pixel coordinates, plus the peak
    heat value. This is what turns a 256px chip hit into a usable pointer at
    the actual feature.
    """
    h, w = chip.shape
    sub = max(16, h // 8)         # sub-patch ~ 1/8 of the chip
    stride = max(4, sub // 4)     # ~32 grid positions per axis

    chip_mean = float(chip.mean())
    chip_var = float(chip.var()) + 1e-8

    gx = np.gradient(chip, axis=1)
    gy = np.gradient(chip, axis=0)
    edge_map = np.sqrt(gx ** 2 + gy ** 2)
    lstd = local_std(chip, size=7)
    tex_mean = float(lstd.mean())
    tex_std = float(lstd.std()) + 1e-8

    ny = max(1, (h - sub) // stride + 1)
    nx = max(1, (w - sub) // stride + 1)
    heat = np.zeros((ny, nx), dtype=np.float32)

    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * stride, ix * stride
            patch = chip[y0:y0 + sub, x0:x0 + sub]
            ep = edge_map[y0:y0 + sub, x0:x0 + sub]
            lp = lstd[y0:y0 + sub, x0:x0 + sub]

            mse_s = float(np.mean((patch - chip_mean) ** 2)) / chip_var
            tex_s = abs(float(lp.mean()) - tex_mean) / tex_std

            strong = ep > float(np.percentile(ep, 80))
            if strong.sum() >= 4:
                angles = np.arctan2(gy[y0:y0 + sub, x0:x0 + sub][strong],
                                    gx[y0:y0 + sub, x0:x0 + sub][strong])
                R = float(abs(np.mean(np.exp(2j * angles))))
                row_a = float(strong.sum(axis=1).max()) / (float(strong.sum()) + 1e-8)
                col_a = float(strong.sum(axis=0).max()) / (float(strong.sum()) + 1e-8)
                edge_s = max(row_a, col_a) * R
            else:
                edge_s = 0.0

            heat[iy, ix] = 0.40 * mse_s + 0.30 * tex_s + 0.30 * edge_s

    heat_s = gaussian_filter(heat.astype(np.float64), sigma=1.5).astype(np.float32)
    lo, hi = float(heat_s.min()), float(heat_s.max())
    if hi - lo < 1e-8:
        return {"fine_bbox": None, "heatmap_peak": 0.0}
    heat_n = (heat_s - lo) / (hi - lo)

    # Region >=65% of peak AND in the top 25% of heat values
    peak_val = float(heat_n.max())
    thr = max(peak_val * 0.65, float(np.percentile(heat_n, 75)))
    hot_ys, hot_xs = np.where(heat_n >= thr)
    if len(hot_ys) == 0:
        idx = np.unravel_index(int(heat_n.argmax()), heat_n.shape)
        hot_ys, hot_xs = np.array([idx[0]]), np.array([idx[1]])

    cy1 = int(hot_ys.min()) * stride
    cy2 = min(h, int(hot_ys.max()) * stride + sub)
    cx1 = int(hot_xs.min()) * stride
    cx2 = min(w, int(hot_xs.max()) * stride + sub)

    return {
        "fine_bbox": [chip_top_x + cx1, chip_top_y + cy1,
                      chip_top_x + cx2, chip_top_y + cy2],
        "heatmap_peak": peak_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  THE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _noop_log(msg: str, level: str = "info") -> None:
    pass


def seed_everything(seed: Optional[int]) -> None:
    """Make a run reproducible. VAE training, augmentation sampling and weight
    init are all stochastic; without a fixed seed two runs on identical inputs
    differ enough to swamp the effect of a config change, which makes an
    iterative tuning harness unable to tell signal from noise."""
    if seed is None:
        return
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def analyze_images(image_paths: List[str],
                   config: Optional[Dict] = None,
                   work_dir: Optional[str] = None,
                   log: Optional[Callable[..., None]] = None,
                   progress: Optional[Callable[[int, int], None]] = None) -> Dict:
    """Run the full Mk19 pipeline over one or more images.

    Args:
        image_paths: scenes to score.
        config:      overrides for DEFAULT_CONFIG.
        work_dir:    scratch dir for chip .npy files. Created if absent; the
                     CALLER owns cleanup (chip_path entries point into it).
        log:         log(msg, level) callback.
        progress:    progress(step, pct) callback.

    Returns a dict with:
        chips        list of scored chip dicts, sorted by combined score desc,
                     each carrying chip metadata, raw + _norm metrics,
                     combined, confidence, rank, is_anomaly, feature_bbox,
                     and (for the top N) fine_bbox / heatmap_peak.
        summary      run metadata: model used, baseline kind, counts, threshold.
    """
    import tempfile

    log = log or _noop_log
    progress = progress or (lambda step, pct: None)
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    seed_everything(cfg.get("seed"))

    chip_size = int(cfg["chip_size"])
    overlap = float(cfg["overlap"])
    percentile = float(cfg["percentile"])
    epochs = int(cfg["epochs"])
    latent_dim = int(cfg["latent_dim"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    warmup = int(cfg["warmup_epochs"])
    trim_frac = float(np.clip(cfg["trim_frac"], 0.0, 0.4))
    max_chips = int(cfg["max_chips"])

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="xenarch_core_")
    chip_dir = os.path.join(work_dir, "chips")

    # ── 1. chips ────────────────────────────────────────────────────────────
    progress(1, 5)
    log(f"Chip extractor: chip_size={chip_size}, overlap={overlap:.0%}")
    all_chips: List[Dict] = []
    for path in image_paths:
        chips = extract_chips(path, chip_dir, chip_size=chip_size,
                              overlap=overlap, max_chips=max_chips)
        # keep chip_id unique across multiple source images
        for c in chips:
            c["chip_id"] = len(all_chips)
            all_chips.append(c)
        log(f"  {Path(path).name}: {len(chips)} chips",
            "success" if chips else "warning")

    if not all_chips:
        raise ValueError("No chips extracted — check image dimensions (need >= chip size).")

    log(f"Total chips: {len(all_chips)}", "success")
    progress(1, 15)
    chip_paths = [c["chip_path"] for c in all_chips]

    # ── 1b. training baseline ───────────────────────────────────────────────
    train_dir = Path(cfg.get("training_dir") or TRAINING_DIR)
    train_imgs = list_training_images(train_dir)
    max_train = int(cfg["max_train_chips"])
    train_chip_paths: List[str] = []
    if train_imgs:
        log(f"Training baseline: {len(train_imgs)} image(s) in {train_dir}")
        train_chip_dir = os.path.join(work_dir, "train_chips")
        budget = max(max_train // len(train_imgs), 20)
        for p in train_imgs:
            tchips = extract_chips(str(p), train_chip_dir, chip_size=chip_size,
                                   overlap=overlap, max_chips=budget)
            train_chip_paths.extend(c["chip_path"] for c in tchips)
            if len(train_chip_paths) >= max_train:
                train_chip_paths = train_chip_paths[:max_train]
                break
        log(f"Baseline chips: {len(train_chip_paths)}",
            "success" if train_chip_paths else "warning")

    use_baseline = len(train_chip_paths) >= 8
    if not use_baseline:
        log(f"No usable training data in {train_dir} — falling back to "
            "self-supervised (trimmed) training on the scene.", "warning")
        train_chip_paths = list(chip_paths)

    # ── 2. model ────────────────────────────────────────────────────────────
    progress(2, 18)
    if HAS_TORCH:
        model_used = ("VAE-Mk19 (PyTorch, natural baseline)" if use_baseline
                      else "VAE-Mk19 (PyTorch, trimmed self)")
    else:
        model_used = "NumPy robust scorer"
    log(f"Fitting {model_used}: train={len(train_chip_paths)} chips, "
        f"score={len(chip_paths)} chips")

    # numpy scorer statistics are fit on the TRAINING distribution
    _ref_sample = [np.load(p) for p in train_chip_paths[:300]]
    np_scorer = NumpyAnomalyScorer(_ref_sample)
    del _ref_sample

    if HAS_TORCH:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StableConvolutionalVAE(latent_dim=latent_dim, input_size=chip_size).to(device)
        optim_ = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_, factor=0.5, patience=2)

        train_paths = list(train_chip_paths)
        keep_idx = np.arange(len(train_chip_paths))

        for epoch in range(epochs):
            # TRIMMED TRAINING: after warmup, drop the top trim_frac highest
            # reconstruction-error chips from this epoch's gradient updates.
            # Even a curated training folder can contain accidental
            # contamination; suspected outliers never get learned in.
            if epoch >= warmup and trim_frac > 0 and len(train_chip_paths) > 10:
                losses = per_chip_recon_mse(model, train_chip_paths, device, batch_size)
                n_keep = max(int(len(train_chip_paths) * (1.0 - trim_frac)), 8)
                keep_idx = np.argsort(losses)[:n_keep]
                train_paths = [train_chip_paths[i] for i in keep_idx]
                if epoch == warmup:
                    log(f"Trimmed training active: excluding top {trim_frac:.0%} "
                        f"({len(train_chip_paths) - n_keep} chips) from gradient updates")

            loader = DataLoader(TorchDataset(train_paths, augment=True),
                                batch_size=batch_size, shuffle=True, num_workers=0)
            model.train()
            kl_w = min(1.0, (epoch + 1) / max(warmup, 1))
            total_loss = 0.0
            nan_hit = False
            for imgs, _ in loader:
                imgs = imgs.to(device)
                optim_.zero_grad()
                recon, mu, logvar = model(imgs)
                loss, _, _ = stable_vae_loss(recon, imgs, mu, logvar, kl_weight=kl_w)
                if torch.isnan(loss):
                    log(f"NaN at epoch {epoch+1}!", "error")
                    nan_hit = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_.step()
                total_loss += loss.item()
            if nan_hit:
                break
            avg = total_loss / max(len(train_paths), 1)
            sched.step(avg)
            log(f"Epoch {epoch+1}/{epochs} [KL={kl_w:.2f}] "
                f"loss={avg:.1f} (train set: {len(train_paths)} chips)")
            progress(2, 18 + int((epoch + 1) / epochs * 34))

        # ── deterministic scoring through mu (no sampling noise) ────────────
        model.eval()
        patch = max(chip_size // 8, 16)

        def encode_and_score(paths):
            errs, mu_list = [], []
            loader = DataLoader(TorchDataset(paths, augment=False),
                                batch_size=batch_size, shuffle=False, num_workers=0)
            with torch.no_grad():
                for imgs, _ in loader:
                    imgs = imgs.to(device)
                    mu, _ = model.encode(imgs)
                    recon = model.decode(mu)
                    errs.extend(patchwise_max_error(imgs, recon, patch=patch).cpu().numpy().tolist())
                    mu_list.append(mu.cpu().numpy())
            return np.array(errs), np.concatenate(mu_list, axis=0)

        log("Scoring baseline chips (deterministic mu-path)…")
        ref_mse, ref_mus = encode_and_score(train_chip_paths)
        if use_baseline:
            log("Scoring scene chips against baseline…")
            scn_mse, scn_mus = encode_and_score(chip_paths)
        else:
            scn_mse, scn_mus = ref_mse, ref_mus   # scene == training set

        # latent distance: robust per-dim z fit on INLIER baseline chips only,
        # so anomalies can't contaminate the reference distribution
        inlier = ref_mus[keep_idx] if len(keep_idx) >= 8 else ref_mus
        l_med = np.median(inlier, axis=0)
        l_mad = np.median(np.abs(inlier - l_med), axis=0) * 1.4826 + 1e-8

        def latent_dist(mus):
            return np.sqrt(np.mean(((mus - l_med) / l_mad) ** 2, axis=1))

        def build_raw(paths, mses, lats, want_bbox):
            out = []
            for i, p in enumerate(paths):
                arr = np.load(p)
                ctx = np_scorer.contextual_score(arr)
                out.append({
                    "mse":          float(mses[i]),
                    "latent":       float(lats[i]),
                    "contextual":   ctx["score"],
                    "gradient":     np_scorer.gradient_score(arr),
                    "edge":         np_scorer.edge_regularity_score(arr),
                    "feature_bbox": ctx["bbox"] if want_bbox else None,
                })
            return out

        raw_scores = build_raw(chip_paths, scn_mse, latent_dist(scn_mus), True)
        ref_raw = (build_raw(train_chip_paths, ref_mse, latent_dist(ref_mus), False)
                   if use_baseline else None)
    else:
        raw_scores = [np_scorer.score(np.load(p)) for p in chip_paths]
        ref_raw = ([np_scorer.score(np.load(p)) for p in train_chip_paths]
                   if use_baseline else None)

    # ── 3. normalize ────────────────────────────────────────────────────────
    # Normalization statistics come from the TRAINING baseline when one is
    # available, so scene scores are absolute ("how far from natural?") rather
    # than relative to whatever happens to be in this scene.
    progress(3, 58)
    log("Robust-normalising scores (median/MAD, "
        + ("stats from training baseline)…" if use_baseline else "self-referenced)…"))
    stats = NumpyAnomalyScorer.fit_norm_stats(ref_raw if use_baseline else raw_scores)
    scored = NumpyAnomalyScorer.apply_norm(raw_scores, stats)
    ref_combined = None
    if use_baseline:
        ref_combined = [r["combined"] for r in
                        NumpyAnomalyScorer.apply_norm(ref_raw, stats)]
    progress(3, 70)

    # ── 4. rank ─────────────────────────────────────────────────────────────
    progress(4, 72)
    scored = compute_confidence(scored, ref_combined=ref_combined)
    thr_src = ref_combined if use_baseline else [s["combined"] for s in scored]
    threshold = float(np.percentile(thr_src, percentile))
    for i, s in enumerate(scored):
        s.update(all_chips[i])
        s["is_anomaly"] = bool(s["combined"] > threshold)
    scored.sort(key=lambda x: x["combined"], reverse=True)
    for i, s in enumerate(scored):
        s["rank"] = i + 1

    n_anomaly = sum(1 for s in scored if s["is_anomaly"])
    n_high = sum(1 for s in scored if s["confidence"] > 0.8)
    log(f"Anomalies: {n_anomaly}  |  High-conf >0.8: {n_high}", "success")
    progress(4, 86)

    # ── 5. stage-2 fine localization on the top-ranked chips ────────────────
    top_n = int(cfg.get("localize_top_n", 5))
    for s in scored:
        s["fine_bbox"] = None
        s["heatmap_peak"] = None
    if top_n > 0:
        log(f"Fine-localising top {min(top_n, len(scored))} chips…")
        for s in scored[:top_n]:
            s.update(localize_within_chip(np.load(s["chip_path"]),
                                          s["top_x"], s["top_y"]))
    progress(5, 95)

    return {
        "chips": scored,
        "summary": {
            "total_chips":    len(scored),
            "n_anomalies":    n_anomaly,
            "n_high_conf":    n_high,
            "top_conf":       round(scored[0]["confidence"], 4) if scored else 0.0,
            "model_used":     model_used,
            "baseline":       "training folder" if use_baseline else "self-supervised",
            "baseline_chips": len(train_chip_paths) if use_baseline else 0,
            "threshold":      threshold,
        },
    }
