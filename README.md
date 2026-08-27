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

Mk19 is current. The detection algorithm lives in **`xenarch_core.py`** and has exactly
one implementation, used by both the web application and the evaluation harness — so the
harness measures the model that actually ships.

In testing over Apollo landing-site imagery, the pipeline ranked the Apollo 11 landing
site hardware among its top anomaly detections without any prior knowledge of the site.
No artifacts from that run are committed here, so it is not reproducible from this
repository; it is recorded as project testing, not as a benchmark.

**No imagery ships with this repository — no training corpus and no evaluation data.**
There are therefore no measured performance numbers in this README. Supply data as
described under [Providing data](#providing-data), then run the harness to produce your
own. The code paths are exercised and working; what is missing is data to point them at.

## Quick start

```bash
pip install -r requirements.txt

# Put natural-geology imagery in ./training data/  (see Providing data)
python xenarch_mk19_script.py     # → http://0.0.0.0:5000
```

With an empty training folder the pipeline still runs, falling back to self-supervised
mode and saying so — but the fixed-baseline behavior that makes scores mean "how far from
natural" only exists once you supply a corpus.

## Repository contents

| File | Role |
| --- | --- |
| `xenarch_core.py` | **The model.** Chip extraction, VAE, trimmed training, the five metrics, normalization, confidence, fine localization. No Flask, no web concerns. |
| `xenarch_mk19_script.py` | Web layer only: upload handling, job tracking, embedded frontend, JSON API. Calls the core. |
| `xenarch_pipeline.py` | Harness-facing `run_pipeline()`. Calls the same core. Also holds a frozen Mk17 scorer for comparison. |
| `eval_harness.py` | Iterative evaluation: per-region separation, tuning vs. held-out splits, stopping criteria. |

## How it works

### 1. Chip extraction

Each input image is normalized to float32 in `[0, 1]` (1st–99th percentile stretch for
GeoTIFF/`.npy`, plain 8-bit scaling for PNG/JPEG) and tiled into square chips, 256 px by
default. Chips overlap by 50%, so a feature straddling a tile seam is not split across
two chips and diluted in both. Near-flat chips (standard deviation below 0.005) are
dropped, and extraction is capped at 500 chips per image.

### 2. Training on natural geology

The VAE trains on imagery in `training data/`, searched recursively for `.tif`, `.tiff`,
`.png`, `.jpg`, `.jpeg`, and `.npy` files. This folder is the curated "all natural
geology" corpus and is the only place the model's notion of normal comes from. Up to 800
chips are sampled from it, budgeted evenly across the available images.

Two mechanisms keep anomalies from being absorbed into that baseline:

- **Trimmed robust training.** After a warmup period, each epoch recomputes per-chip
  reconstruction error and excludes the worst-reconstructing 8% of chips from gradient
  updates. Even a curated corpus can contain accidental contamination; suspected
  outliers never get learned in. This is the fix for the failure mode where the VAE
  memorized a lander and then reported it as normal.
- **Augmentation.** Random 90° rotations and horizontal flips force the model to learn
  geology *statistics* rather than memorize individual chips.

The scene is then scored against this fixed baseline. Normalization statistics, the
anomaly threshold, and confidence z-scores all come from the training distribution, so a
score means "how far from known-natural" rather than "how weird relative to this
particular scene."

If the training folder is missing or holds fewer than 8 usable chips, the pipeline falls
back to self-supervised trimmed training on the scene itself and says so in the job log
and in the result summary's `baseline` field.

### 3. Scoring

Chips are encoded deterministically through the latent mean `mu` — no sampling noise
enters the ranking. Five metrics are computed per chip and combined:

| Metric | Weight | What it measures |
| --- | --- | --- |
| `mse` | 0.30 | **Patch-wise maximum** reconstruction error. Error is pooled over local windows and the maximum taken, so a small artifact dominates its chip instead of being averaged away by 65k background pixels. |
| `edge` | 0.25 | Orientation-invariant edge regularity. Row/column alignment of strong edges plus FFT angular-spectrum concentration — natural terrain has an isotropic spectrum, while a straight edge at *any* angle concentrates energy in a narrow angular band. |
| `contextual` | 0.20 | Compact locally-deviant region, **bright or dark**, plus texture-outlier fraction. The two-sided test catches shadowed hardware as well as sunlit hardware. |
| `latent` | 0.15 | Robust per-dimension z-distance in latent space, with median/MAD statistics fit on inlier baseline chips only, so anomalies cannot contaminate the reference distribution. |
| `gradient` | 0.10 | Local gradient irregularity relative to a smoothed gradient field. |

Edge regularity carries real weight on the reasoning that straight lines at arbitrary
orientation are the strongest geology-versus-technology discriminator available in a
single grayscale chip. That weighting is a design judgment, not yet a measured one —
confirming it against real labeled data is a first job for the harness.

### 4. Normalization and confidence

Each metric is converted to a robust z-score (median/MAD, fit on the training baseline)
and squashed through a sigmoid, rather than min-max scaled. Min-max let a single extreme
chip compress everything else into a narrow band; median/MAD does not.

Confidence is the sigmoid of the combined score's robust z, **centered at z = 2**. A chip
two robust sigmas above the baseline median lands at 0.5; four sigmas reaches about 0.88.
The practical consequence is that the top-ranked chip is not automatically "confident" —
if a scene contains nothing but natural geology, every chip in it scores low. That
property is what makes the output a review queue rather than a forced ranking.

A chip is flagged `is_anomaly` when its combined score exceeds the 92nd percentile of the
baseline distribution.

### 5. Fine localization

The top-ranked chips get a stage-2 dense sliding-window heatmap scoring every sub-patch
on background deviation, texture outlierness, and directional edge regularity. This
returns `fine_bbox` in **image pixel coordinates** — a box around the feature itself
rather than the 256 px chip containing it, so a reviewer is pointed at the feature
instead of a 65,536 px² tile.

### 6. Determinism

Runs are seeded by default (`config["seed"]`, default `0`). VAE weight init, augmentation
sampling, and batch shuffling are all stochastic, and unseeded runs on identical inputs
differ enough to obscure the effect of a config change — which would leave an iterative
tuning harness unable to separate a real improvement from noise. Set `seed` to `null` for
nondeterministic runs.

## Providing data

Nothing here ships with imagery. Two separate things need supplying.

### Training corpus — required for real use

Put natural-geology imagery in `training data/` (or point `XENARCH_TRAINING_DIR` /
`config["training_dir"]` elsewhere). Subfolders are searched recursively;
`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.npy` are accepted.

This corpus defines "normal," so detection quality is bounded by how well it represents
the terrain you intend to scan. Match illumination geometry, resolution, and terrain type
to the scenes you will score — a baseline built from one and applied to the other will
flag the mismatch itself as anomalous. Trimmed training tolerates incidental
contamination, but the corpus should be natural geology by intent.

### Evaluation data — required only for the harness

`eval_harness.py` needs labeled regions. Without them it exits with a message describing
the layout rather than a traceback.

```
eval/regions.json
eval/<region_id>/ground_truth.json
eval/<region_id>/<scene image>
```

`eval/regions.json`:

```json
{"regions": [{"region_id":    "apollo_11",
              "split":        "tuning",
              "anomaly_type": "lander",
              "chip_size":    256}]}
```

`split` is `tuning` or `held-out`; `anomaly_type` is a free-form label used only to group
the per-type summary; `chip_size` is an optional per-region override.

`eval/<region_id>/ground_truth.json`:

```json
{"image_path": "eval/apollo_11/scene.tif",
 "low_confidence": false,
 "zones": [{"label": "target",         "bbox": [x1, y1, x2, y2]},
           {"label": "false_positive", "bbox": [x1, y1, x2, y2]},
           {"label": "background",     "bbox": [0, 0, w, h]}]}
```

Two details matter when drawing zones:

**Order them target/false-positive first.** A chip is labeled by whether its center falls
inside a zone, first match winning, so a catch-all `background` zone must come last.

**Size them to containment, not to the feature.** A chip of width `chip_size` centered at
`c` spans `[c - chip_size/2, c + chip_size/2]`, so the chips that *fully contain* a
feature spanning `[f1, f2]` are exactly those with center in
`[f2 - chip_size/2, f1 + chip_size/2]`. Drawing the zone tightly around the feature
instead under-labels — chips that genuinely contain it get scored as background. Drawing
it wider over-labels, tagging chips that hold only a clipped corner as `target`, which
drags target confidence down and understates separation. Either way the harness ends up
measuring the labeling rather than the model.

A region needs at least one `target` and one `false_positive` chip to produce a
separation score. Good `false_positive` zones are the natural features that are *hard* —
bright-rimmed fresh craters, boulder fields, high-contrast scarps — not empty terrain.

## Running the web app

### Dependencies

```bash
pip install -r requirements.txt
```

`torch` and `rasterio` are optional. Without `torch`, scoring falls back to a NumPy robust
scorer that substitutes a smooth-background residual for VAE reconstruction error and a
statistical fingerprint distance for the latent metric; the pipeline runs, but detection
quality drops. Without `rasterio`, image loading falls back to Pillow and GeoTIFF metadata
is ignored.

### Local and production

```bash
python xenarch_mk19_script.py          # dev, → http://0.0.0.0:5000
gunicorn xenarch_mk19_script:app       # production; see Procfile
```

**Run a single worker.** Job state lives in process memory, so a client polling
`/api/progress` against a second worker gets a 404 for a job that is running fine
elsewhere. The `Procfile` pins `--workers 1 --threads 4`. Finished jobs (and their chip
scratch directories) are evicted after `XENARCH_JOB_TTL` seconds or once more than
`XENARCH_MAX_JOBS` are retained.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `XENARCH_TRAINING_DIR` | `./training data` | Natural-geology corpus the VAE trains on. |
| `PORT` / `HOST` | `5000` / `0.0.0.0` | Bind address. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Left permissive for first deployment; the server logs a warning at startup while it is `*`. Set it before exposing the API. |
| `XENARCH_JOB_TTL` | `3600` | Seconds to retain finished jobs. |
| `XENARCH_MAX_JOBS` | `32` | Max jobs retained in memory. |
| `FLASK_DEBUG` | `0` | Set to `1` for Flask debug mode. |

## HTTP API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Embedded frontend: drag-and-drop upload, live job log, ranked detections. |
| `/api/status` | GET | Health check: version, torch/rasterio availability, training dir and image count, active baseline, combined weights, job counts. |
| `/api/analyze` | POST | Multipart upload. `files[]` = one or more images; optional `config` form field with a JSON config object. Returns `{"job_id": ...}` and runs on a background thread. |
| `/api/progress/<job_id>` | GET | Step, percent, last 50 log lines, done flag, error. |
| `/api/results/<job_id>` | GET | Ranked detections plus summary. 202 while running, 500 if the job failed. |

### Job configuration

All keys optional; these are the defaults.

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
| `max_chips` | `500` | Cap on chips extracted per scene. |
| `localize_top_n` | `5` | How many top chips get stage-2 fine localization. |
| `training_dir` | env default | Per-job override of the training corpus. |
| `seed` | `0` | RNG seed; `null` for nondeterministic. |
| `engine` | `"mk19"` | `"mk17"` selects the frozen legacy scorer, for comparison only. |

Training stability comes from gradient-norm clipping at 1.0, `logvar` clamping to
`[-10, 10]`, small-gain Xavier init on the latent heads, `ReduceLROnPlateau` scheduling,
and a NaN guard that aborts training rather than propagating garbage.

## Evaluation

`eval_harness.py` measures whether the detector separates real targets from known-hard
natural false positives, on a tuning split and a held-out split.

```bash
python eval_harness.py --iteration 1
python eval_harness.py --iteration 2 --config my_config.json
python eval_harness.py --status
```

The core metric is **separation**: mean confidence over `target` chips minus mean
confidence over `false_positive` chips, per region. The harness reports per-region
separation, aggregate mean and worst-case per split, a per-metric breakdown showing which
metric is carrying the discrimination, and a summary by anomaly type.

It stops on one of three conditions:

- **Success** — both tuning and held-out mean separation exceed the threshold (default 0.30).
- **Overfitting plateau** — tuning separation improved while held-out stagnated across
  `overfit_patience` iterations (default 2). The harness explicitly tells you not to
  proceed with the current changes.
- **Iteration cap** — default 6.

Results land in `results/iteration_N/` as `report.json` plus per-region `chip_labels.csv`,
with a cumulative `results/progress_log.csv` recording every iteration's weights and
separation figures.

The tuning/held-out split is the point of the exercise. Metric weights are hand-set, and
hand-tuning five weights against a handful of regions overfits easily; the held-out split
and the plateau check are what make an improvement believable.

## Known limitations

- **No measured performance.** No training corpus and no evaluation data ship here, so
  nothing in this README reports detection accuracy. The metric weights are reasoned
  design choices that have not been validated against labeled real imagery.
- **The baseline VAE retrains per scene.** The harness trains from scratch for every
  region against the same baseline corpus, which is roughly *N*× more compute than needed
  for *N* regions. The fix is to train once, persist the weights and the baseline
  normalization statistics, and reuse them across scenes — worth doing before the corpus
  grows.
- **Single-process job state**, as described under Running the web app.
- **CORS defaults to `*`.** Convenient for first deployment, wrong for anything exposed.
- **The legacy Mk17 engine tiles without overlap**, producing far fewer chips whose
  centers frequently miss labeled zones entirely. `{"engine": "mk17"}` runs it for
  comparison, but expect regions to come back unscoreable rather than merely worse.

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

A third property is structural: **the harness must measure the shipping model.** Keeping
one implementation in `xenarch_core.py` is what will make the separation numbers mean
anything once there is data to produce them. When the web app and the harness drifted
apart, the harness was scoring code nobody ran.
