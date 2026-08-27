# Xenarch

**Unsupervised anomaly detection for planetary-surface imagery.**

Xenarch trains a convolutional variational autoencoder (VAE) on a corpus of natural
geology and then scores new spacecraft imagery by asking a single question: *how well
can this be reconstructed as known natural terrain?* Anything the model cannot explain
as geology is ranked highly and surfaced for human review.

The model is never told what an anomaly looks like. There are no anomaly labels, no
positive class, and no training examples of artificial structures. The only supervision
is the choice of what goes into the natural-geology training folder.

## Status

Current production entry point is `xenarch_mk19_script.py` (Mk19), a self-contained
Flask application with an embedded web frontend.

In testing over Apollo landing-site imagery, the pipeline ranked the Apollo 11 landing
site hardware among its top anomaly detections without any prior knowledge of the site.
Note that no evaluation artifacts from that run are committed to this repository — the
result is recorded here from project testing, not reproducible from files currently in
the repo. See [Evaluation](#evaluation) for the harness that produces reproducible
numbers, and [Known gaps](#known-gaps) for what is missing.

## Repository contents

| File | Role |
| --- | --- |
| `xenarch_mk19_script.py` | Mk19 production pipeline + Flask API + embedded web UI. The current model. |
| `xenarch_pipeline.py` | Importable, Flask-free pipeline used by the evaluation harness. Still on Mk17-era scoring — see [Known gaps](#known-gaps). |
| `eval_harness.py` | Iterative evaluation harness: scores labeled regions, tracks tuning vs. held-out separation, applies stopping criteria. |

## How it works

The Mk19 pipeline runs in five stages.

### 1. Chip extraction

Each input image is normalized to float32 in `[0, 1]` (1st–99th percentile stretch for
GeoTIFF/`.npy`, plain 8-bit scaling for PNG/JPEG) and tiled into square chips, 256 px by
default. Chips overlap by 50%, so a feature straddling a tile seam is not split across
two chips and diluted in both. Near-flat chips (standard deviation below 0.005) are
dropped, and extraction is capped at 500 chips per image.

### 2. Training on natural geology

The VAE trains on imagery in the project's `training data/` folder, searched recursively
for `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, and `.npy` files. This folder is the
curated "all natural geology" corpus and is the only place the model's notion of normal
comes from. Up to 800 chips are sampled from it, budgeted evenly across the available
images.

Two mechanisms keep anomalies from being absorbed into that baseline:

- **Trimmed robust training.** After a warmup period, each epoch recomputes per-chip
  reconstruction error and excludes the worst-reconstructing 8% of chips from gradient
  updates. Even a curated corpus can contain accidental contamination; suspected
  outliers never get learned in. This is the fix for the failure mode where the VAE
  memorized a lander and then reported it as normal.
- **Augmentation.** Random 90° rotations and horizontal flips force the model to learn
  geology *statistics* rather than memorize individual chips.

The uploaded scene is then scored against this fixed baseline. Normalization statistics,
the anomaly threshold, and confidence z-scores all come from the training distribution,
so a score means "how far from known-natural" rather than "how weird relative to this
particular scene."

If the training folder is missing or holds fewer than 8 usable chips, the pipeline falls
back to self-supervised trimmed training on the uploaded scene itself and says so in the
job log and in the result summary's `baseline` field.

### 3. Scoring

Chips are encoded deterministically through the latent mean `mu` — no sampling noise
enters the ranking. Five metrics are computed per chip and combined:

| Metric | Weight | What it measures |
| --- | --- | --- |
| `mse` | 0.30 | **Patch-wise maximum** reconstruction error. Error is pooled over local windows and the maximum is taken, so a small artifact dominates its chip instead of being averaged away by 65k background pixels. |
| `edge` | 0.25 | Orientation-invariant edge regularity. Combines row/column alignment of strong edges with FFT angular-spectrum concentration — natural terrain has an isotropic spectrum, while a straight edge at *any* angle concentrates energy in a narrow angular band. |
| `contextual` | 0.20 | Compact locally-deviant region, **bright or dark**, plus texture-outlier fraction. The two-sided test catches shadowed hardware as well as sunlit hardware. |
| `latent` | 0.15 | Robust per-dimension z-distance in latent space, with median/MAD statistics fit on inlier baseline chips only so anomalies cannot contaminate the reference distribution. |
| `gradient` | 0.10 | Local gradient irregularity relative to a smoothed gradient field. |

Edge regularity carries real weight because straight lines at arbitrary orientation are
the strongest geology-versus-technology discriminator available in a single grayscale
chip.

### 4. Normalization and confidence

Each metric is converted to a robust z-score (median/MAD, fit on the training baseline)
and squashed through a sigmoid, rather than min-max scaled. Min-max normalization let a
single extreme chip compress everything else into a narrow band; median/MAD does not.

Confidence is the sigmoid of the combined score's robust z, **centered at z = 2**. A chip
two robust sigmas above the baseline median lands at 0.5; four sigmas reaches about 0.88.
The practical consequence is that the top-ranked chip is not automatically "confident" —
if a scene contains nothing but natural geology, every chip in it scores low. That
property is what makes the output usable as a review queue instead of a forced ranking.

A chip is flagged `is_anomaly` when its combined score exceeds the 92nd percentile of the
baseline distribution.

### 5. Output

The top 12 chips are returned with base64 PNG thumbnails, a feature bounding box within
the chip, per-metric normalized values, and calibrated confidence.

## Running it

### Dependencies

```bash
pip install numpy scipy pillow flask flask-cors loguru
pip install torch        # optional, enables the VAE — strongly recommended
pip install rasterio     # optional, GeoTIFF support
```

Both optional dependencies degrade gracefully. Without `torch`, scoring falls back to a
NumPy-only robust scorer that substitutes a smooth-background residual for the VAE
reconstruction error and a statistical fingerprint distance for the latent metric — the
pipeline still runs, but detection quality drops. Without `rasterio`, image loading falls
back to Pillow and GeoTIFF metadata is ignored.

### Local development

```bash
python xenarch_mk19_script.py
# → http://0.0.0.0:5000
```

### Production

```bash
gunicorn xenarch_mk19_script:app
```

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `XENARCH_TRAINING_DIR` | `./training data` | Natural-geology corpus the VAE trains on. |
| `PORT` | `5000` | HTTP port. |
| `HOST` | `0.0.0.0` | Bind address. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Tighten this before any real deployment. |
| `FLASK_DEBUG` | `0` | Set to `1` for Flask debug mode. |

## HTTP API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Embedded web frontend: drag-and-drop upload, live job log, ranked detections. |
| `/api/status` | GET | Health check. Reports version, whether torch/rasterio are available, the training directory, and how many training images were found. |
| `/api/analyze` | POST | Multipart upload. `files[]` = one or more images; optional `config` form field carrying a JSON config object. Returns `{"job_id": ...}` and runs the analysis on a background thread. |
| `/api/progress/<job_id>` | GET | Current step, percent complete, last 50 log lines, done flag, error. |
| `/api/results/<job_id>` | GET | Ranked detections plus a summary. Returns 202 while still running, 500 if the job failed. |

### Job configuration

All keys are optional; these are the defaults.

| Key | Default | Meaning |
| --- | --- | --- |
| `chip_size` | `256` | Tile size in pixels. |
| `overlap` | `0.5` | Fractional chip overlap. |
| `percentile` | `92` | Anomaly threshold percentile against the baseline. |
| `epochs` | `20` | Training epochs. |
| `latent_dim` | `56` | VAE latent dimensionality. |
| `batch_size` | `4` | Training batch size. |
| `lr` | `0.0005` | Adam learning rate. |
| `warmup_epochs` | `3` | KL warmup length; trimmed training begins after this. |
| `trim_frac` | `0.08` | Fraction of worst-reconstructing chips excluded from gradient updates, clamped to `[0, 0.4]`. |
| `max_train_chips` | `800` | Cap on baseline chips sampled from the training folder. |
| `training_dir` | `XENARCH_TRAINING_DIR` | Per-job override of the training corpus. |

Training stability is handled with gradient-norm clipping at 1.0, `logvar` clamping to
`[-10, 10]`, small-gain Xavier initialization on the latent heads, `ReduceLROnPlateau`
scheduling, and a NaN guard that aborts training rather than propagating garbage.

## Evaluation

`eval_harness.py` measures whether the detector actually separates real targets from
known-hard false positives, on a tuning split and a held-out split.

```bash
python eval_harness.py --iteration 1
python eval_harness.py --iteration 2 --config my_config.json
python eval_harness.py --status
```

The core metric is **separation**: mean confidence over `target` chips minus mean
confidence over `false_positive` chips, per region. The harness reports per-region
separation, aggregate mean and worst-case separation per split, a per-metric breakdown
for diagnosis, and a summary by anomaly type.

It stops on one of three conditions:

- **Success** — both tuning and held-out mean separation exceed the threshold (default 0.30).
- **Overfitting plateau** — tuning separation improved while held-out separation stagnated
  across `overfit_patience` iterations (default 2). The harness explicitly tells you not to
  proceed with the current changes.
- **Iteration cap** — default 6.

Results land in `results/iteration_N/` as `report.json` plus per-region `chip_labels.csv`,
with a cumulative `results/progress_log.csv` recording every iteration's weights and
separation figures.

### Expected inputs

The harness reads:

- `eval/regions.json` — list of regions, each with `region_id`, `split`
  (`tuning` | `held-out`), `anomaly_type`, and an optional `chip_size` override.
- `eval/<region_id>/ground_truth.json` — `image_path` plus `zones`, each a bounding box
  labeled `target`, `false_positive`, or `background`. Chips are assigned by center point,
  first matching zone wins.

## Known gaps

These are real and worth knowing before you rely on any of this:

- **No evaluation data is committed.** The `eval/` tree, the `training data/` corpus, and
  the `results/` output directory are all absent from the repository. The harness cannot
  run as checked out, and the Apollo 11 result above cannot currently be reproduced from
  this repo alone.
- **The harness evaluates the wrong model.** `eval_harness.py` imports `run_pipeline` from
  `xenarch_pipeline.py`, which still carries Mk17-era scoring: non-overlapping chips,
  min-max normalization, an `mse`/`density`-dominated weight vector
  (0.60 / 0.30 / 0.05 / 0.05 / 0.00), and a VAE path that is off by default and lacks
  trimmed training. Numbers it produces do **not** describe Mk19. Bringing the Mk19 scorer
  behind the harness's interface is the highest-value next change.
- **Missing packaging files.** `requirements.txt` and the `Procfile` referenced in the Mk19
  module docstring are not in the repository. Install dependencies with the pip commands
  above until they are added.
- **Fine localization only exists in the old pipeline.** `xenarch_pipeline.py` has a
  stage-2 `localize_within_chip` step that produces a pixel-coordinate `fine_bbox` via a
  dense sliding-window heatmap. Mk19 returns only a coarse chip-relative `feature_bbox`.
- **In-memory job state.** `PROGRESS` and `RESULTS` are process-local dictionaries that
  are never evicted. A multi-worker gunicorn deployment will lose jobs across workers, and
  a long-lived single worker will grow without bound.
- **CORS defaults to `*`.** Convenient for a first deployment, wrong for anything exposed.

## Design intent

Two properties matter more than raw detection rate, and both are deliberate:

**The model must not learn the thing it is looking for.** Trimmed training, augmentation,
inlier-only latent statistics, and the raised KL weight all exist to keep rare artificial
structures out of the learned notion of "natural." A VAE that reconstructs a lander
perfectly reports nothing.

**Confidence must be able to say "nothing here."** Because scores are z-scored against a
fixed natural baseline rather than against the current scene, a scene of ordinary terrain
produces uniformly low confidence instead of a confidently-ranked list of ordinary rocks.
The output is a review queue for humans, and a review queue that cries wolf on every scene
is worse than none.
