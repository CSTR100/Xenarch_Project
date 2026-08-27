"""
Xenarch Mk19 — Production Web Edition
=====================================
Unsupervised planetary-surface technosignature detection: a VAE is trained on
natural geology, and anything it cannot reconstruct as natural is surfaced for
human review.

The detection algorithm lives in `xenarch_core.py`, NOT here. This module is
the web layer: upload handling, job tracking, the embedded frontend, and the
JSON API. `xenarch_pipeline.py` calls the same core, so `eval_harness.py`
measures exactly the model this server runs.

What Mk19 changed, and why (all implemented in xenarch_core):
  1. TRIMMED ROBUST TRAINING — after a warmup, each epoch excludes the top-k%
     highest reconstruction-error chips from gradient updates, so anomalies are
     never absorbed into the "natural geology" baseline. This is the fix for
     the core contamination problem (the VAE previously memorized the lander).
  2. AUGMENTATION — random flips / 90-degree rotations force the VAE to learn
     geology statistics instead of memorizing individual chips.
  3. DETERMINISTIC SCORING — chips are scored through the latent mean (mu),
     so there is no sampling noise in the ranking.
  4. PATCH-WISE MAX ERROR — reconstruction error is pooled over local windows
     and the MAX taken, so a small artifact dominates its chip instead of being
     averaged away by 65k background pixels.
  5. LATENT MAHALANOBIS DISTANCE — robust per-dimension z-distance in latent
     space, with statistics fit on inlier chips only.
  6. TWO-SIDED CONTEXTUAL SCORE — detects dark compact features (shadowed
     hardware) as well as bright ones.
  7. ORIENTATION-INVARIANT EDGE REGULARITY — FFT angular-spectrum concentration
     catches straight edges at ANY angle; weighted up from 5% to 25%.
  8. ROBUST NORMALIZATION — median/MAD z-scores + sigmoid instead of min-max,
     so one extreme chip can't compress the rest of the distribution.
  9. CALIBRATED CONFIDENCE — sigmoid of the robust z of the combined score;
     stays LOW when nothing in the scene is genuinely anomalous.
 10. 50% CHIP OVERLAP — features straddling chip boundaries are no longer
     split and diluted.
 11. TRAINING FOLDER BASELINE — the VAE trains on imagery in `training data/`
     (curated natural geology; searched recursively; TIF/PNG/JPG/NPY). Scenes
     are scored against that FIXED baseline, so scores mean "how far from
     known-natural" rather than "how weird relative to this scene". Override
     with XENARCH_TRAINING_DIR or per-job via config["training_dir"]. If the
     folder is missing or empty, the pipeline falls back to self-supervised
     trimmed training.
 12. STAGE-2 FINE LOCALIZATION — the top-ranked chips get a dense sliding-window
     heatmap, returning a pixel-coordinate box around the feature itself rather
     than just the 256px chip that contains it.

Local dev:
    pip install -r requirements.txt
    python xenarch_mk19_script.py

Production (via gunicorn — see Procfile):
    gunicorn xenarch_mk19_script:app

Environment variables (all optional):
    PORT                 — HTTP port (default 5000)
    HOST                 — bind address (default 0.0.0.0)
    ALLOWED_ORIGINS      — comma-separated CORS origins. Defaults to * for easy
                           first deployment; set it before exposing the server.
    XENARCH_TRAINING_DIR — natural-geology baseline folder
    XENARCH_JOB_TTL      — seconds to retain finished jobs (default 3600)
    XENARCH_MAX_JOBS     — max jobs retained in memory (default 32)

NOTE ON DEPLOYMENT: job state lives in this process's memory. Run a SINGLE
worker (the Procfile pins `--workers 1`); with more, a client polling
/api/progress will hit a worker that has never heard of its job.
"""

import os
import sys
import json
import base64
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from loguru import logger
from PIL import Image

import xenarch_core as core
from xenarch_core import (
    CHIP_SIZE,
    COMBINED_WEIGHTS,
    METRIC_KEYS,
    TRAINING_DIR,
    HAS_RASTERIO,
    HAS_TORCH,
    list_training_images,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

if not HAS_TORCH:
    logger.warning("PyTorch not available — using numpy-only anomaly scoring")
if not HAS_RASTERIO:
    logger.warning("rasterio not available — falling back to Pillow for image I/O")

# ─────────────────────────────────────────────────────────────────────────────
# JOB STATE
# ─────────────────────────────────────────────────────────────────────────────
# In-memory and therefore single-process. Entries are evicted by age and by
# count so a long-running server doesn't grow without bound; each job also owns
# a temp dir that must be removed when the job is dropped.

PROGRESS: Dict[str, Dict] = {}
RESULTS: Dict[str, Dict] = {}
JOB_DIRS: Dict[str, str] = {}
_JOBS_LOCK = threading.Lock()

JOB_TTL = float(os.environ.get("XENARCH_JOB_TTL", 3600))    # seconds
MAX_JOBS = int(os.environ.get("XENARCH_MAX_JOBS", 32))


def _evict_jobs() -> None:
    """Drop finished jobs older than JOB_TTL, then the oldest jobs beyond
    MAX_JOBS. Running jobs are never evicted."""
    now = time.time()
    with _JOBS_LOCK:
        stale = [jid for jid, p in PROGRESS.items()
                 if p.get("done") and now - p.get("created", now) > JOB_TTL]
        finished = sorted((jid for jid, p in PROGRESS.items() if p.get("done")),
                          key=lambda j: PROGRESS[j].get("created", 0))
        overflow = finished[:max(0, len(PROGRESS) - MAX_JOBS)]
        for jid in set(stale) | set(overflow):
            PROGRESS.pop(jid, None)
            RESULTS.pop(jid, None)
            d = JOB_DIRS.pop(jid, None)
            if d:
                shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

if CORS_ORIGINS == "*":
    logger.warning(
        "ALLOWED_ORIGINS is '*' — any website can call this API from a "
        "visitor's browser. Set ALLOWED_ORIGINS before exposing this server.")


# ─────────────────────────────────────────────────────────────────────────────
# THUMBNAIL
# ─────────────────────────────────────────────────────────────────────────────

def chip_to_b64(chip_path: str, size: int = 256) -> str:
    try:
        arr = np.load(chip_path)
    except Exception:
        arr = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# JOB RUNNER  (thin adapter over the core engine)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(job_id: str, image_paths: List[str], config: Dict, tmp_dir: str):
    """Run the core pipeline, streaming progress into PROGRESS[job_id] and
    packaging the top detections into RESULTS[job_id]."""
    PROGRESS[job_id] = {"step": 0, "pct": 0, "logs": [], "done": False,
                        "error": None, "created": time.time()}

    def log(msg, level="info"):
        PROGRESS[job_id]["logs"].append({
            "t": time.strftime("%H:%M:%S.") + f"{int(time.time()*1000)%1000:03d}",
            "msg": msg, "level": level})
        getattr(logger, level if level in ("info", "warning", "error") else "info")(msg)

    def progress(step, pct):
        PROGRESS[job_id]["step"] = step
        PROGRESS[job_id]["pct"] = pct

    try:
        out = core.analyze_images(image_paths, config, work_dir=tmp_dir,
                                  log=log, progress=progress)
        scored = out["chips"]

        log("Generating thumbnails…")
        detections = []
        for rank, s in enumerate(scored[:min(12, len(scored))], 1):
            detections.append({
                "rank":        rank,
                "chipName":    Path(s["chip_path"]).stem,
                "confidence":  round(s["confidence"], 4),
                "score":       round(s["combined"], 4),
                "source":      s.get("source", ""),
                "imgDataURI":  chip_to_b64(s["chip_path"]),
                "featureBbox": s.get("feature_bbox"),
                "fineBbox":    s.get("fine_bbox"),
                "metrics":     {k: round(s.get(f"{k}_norm", 0), 3) for k in METRIC_KEYS},
            })

        if detections:
            log(f"Rank 1: {detections[0]['chipName']}  "
                f"conf={detections[0]['confidence']:.4f}", "success")
        progress(5, 100)
        log("Analysis complete ✓", "success")

        RESULTS[job_id] = {
            "summary": out["summary"],
            "detections": detections,
            "csv_rows": [{k: d[k] for k in
                          ("chipName", "confidence", "score", "source", "metrics")}
                         for d in detections],
        }
    except Exception as exc:
        logger.error(traceback.format_exc())
        PROGRESS[job_id]["error"] = str(exc)
    finally:
        PROGRESS[job_id]["done"] = True
        _evict_jobs()



# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED FRONTEND  (served at GET /)
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>XENARCH · Planetary Technosignature Detection · Mk19</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap" rel="stylesheet" />
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #04060d;
    --surface:   #090e1a;
    --panel:     #0d1526;
    --border:    #1a2744;
    --accent:    #00e5ff;
    --accent2:   #ff4f00;
    --dim:       #3a5080;
    --text:      #c8daf5;
    --textlo:    #4a6080;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow Condensed', sans-serif;
    --glow:      0 0 18px rgba(0,229,255,.35);
    --glow2:     0 0 18px rgba(255,79,0,.35);
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); }

  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px,
      rgba(0,0,0,.09) 3px, rgba(0,0,0,.09) 4px);
  }

  .shell { max-width: 1280px; margin: 0 auto; padding: 0 24px 80px; }

  header {
    display: flex; align-items: center; gap: 20px;
    padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 36px;
  }
  .logo-mark {
    width: 44px; height: 44px; border: 2px solid var(--accent); border-radius: 4px;
    display: grid; place-items: center; box-shadow: var(--glow); position: relative; flex-shrink: 0;
  }
  .logo-mark::after {
    content: ''; position: absolute; inset: 5px; border: 1px solid var(--accent);
    border-radius: 2px; opacity: .5;
  }
  .logo-cross {
    width: 16px; height: 16px;
    background: linear-gradient(var(--accent), var(--accent)) 50% 0/2px 100%,
                linear-gradient(var(--accent), var(--accent)) 0 50%/100% 2px;
    background-color: transparent;
  }
  .logo-text h1 {
    font-family: var(--mono); font-size: 22px; letter-spacing: .18em;
    color: var(--accent); text-shadow: var(--glow);
  }
  .logo-text p {
    font-size: 11px; letter-spacing: .25em; color: var(--dim);
    text-transform: uppercase; margin-top: 2px;
  }
  .header-badge {
    margin-left: auto; font-family: var(--mono); font-size: 10px; color: var(--dim);
    letter-spacing: .1em; text-align: right; line-height: 1.8;
  }
  .header-badge span { color: var(--accent); }

  .main-grid { display: grid; grid-template-columns: 320px 1fr; gap: 24px; }
  @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }

  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 24px; }
  .panel-title {
    font-family: var(--mono); font-size: 11px; letter-spacing: .18em; color: var(--dim);
    text-transform: uppercase; margin-bottom: 18px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }

  #dropzone {
    border: 2px dashed var(--border); border-radius: 6px; padding: 36px 16px;
    text-align: center; cursor: pointer; transition: border-color .2s, background .2s; position: relative;
  }
  #dropzone:hover, #dropzone.drag { border-color: var(--accent); background: rgba(0,229,255,.04); }
  #dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  #dropzone .dz-icon { font-size: 28px; margin-bottom: 10px; opacity: .5; }
  #dropzone .dz-label { font-size: 13px; color: var(--textlo); line-height: 1.6; }
  #dropzone .dz-label span { color: var(--accent); }

  #file-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .file-tag {
    background: rgba(0,229,255,.07); border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 10px; font-size: 11px; font-family: var(--mono); color: var(--text);
    display: flex; align-items: center; justify-content: space-between;
  }
  .file-tag button { background: none; border: none; color: var(--dim); cursor: pointer; font-size: 13px; }
  .file-tag button:hover { color: var(--accent2); }

  .param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
  .param-block label {
    display: block; font-size: 10px; letter-spacing: .12em; color: var(--dim);
    margin-bottom: 5px; text-transform: uppercase;
  }
  .param-block input, .param-block select {
    width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); font-family: var(--mono); font-size: 13px; padding: 7px 10px;
    outline: none; transition: border-color .15s;
  }
  .param-block input:focus, .param-block select:focus { border-color: var(--accent); }

  #run-btn {
    margin-top: 22px; width: 100%; padding: 13px; background: transparent;
    border: 2px solid var(--accent); border-radius: 4px; color: var(--accent);
    font-family: var(--mono); font-size: 14px; letter-spacing: .18em; cursor: pointer;
    text-transform: uppercase; box-shadow: var(--glow);
    transition: background .2s, color .2s, box-shadow .2s; position: relative; overflow: hidden;
  }
  #run-btn:hover:not(:disabled) {
    background: var(--accent); color: var(--bg); box-shadow: 0 0 28px rgba(0,229,255,.6);
  }
  #run-btn:disabled { opacity: .4; cursor: not-allowed; }

  .right-col { display: flex; flex-direction: column; gap: 20px; }

  #progress-panel { display: none; }
  .progress-steps {
    display: flex; gap: 0; margin-bottom: 20px; border: 1px solid var(--border);
    border-radius: 4px; overflow: hidden;
  }
  .step-item {
    flex: 1; padding: 10px 6px; text-align: center; font-size: 10px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--dim); border-right: 1px solid var(--border);
    transition: background .3s, color .3s;
  }
  .step-item:last-child { border-right: none; }
  .step-item.active { background: rgba(0,229,255,.12); color: var(--accent); }
  .step-item.done   { background: rgba(0,229,255,.06); color: var(--text); }

  .prog-bar-outer { height: 4px; background: var(--border); border-radius: 2px; margin-bottom: 16px; overflow: hidden; }
  .prog-bar-inner {
    height: 100%; background: var(--accent); border-radius: 2px;
    box-shadow: var(--glow); width: 0%; transition: width .4s ease;
  }

  #log-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
    padding: 12px 14px; height: 160px; overflow-y: auto; font-family: var(--mono);
    font-size: 11px; line-height: 1.9; color: var(--textlo);
  }
  #log-box .log-ok   { color: #40e090; }
  #log-box .log-warn { color: #ffb347; }
  #log-box .log-err  { color: #ff4f4f; }

  #results-panel { display: none; }
  .summary-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .summary-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
    padding: 14px; text-align: center;
  }
  .summary-card .sc-val {
    font-family: var(--mono); font-size: 26px; color: var(--accent);
    text-shadow: var(--glow); line-height: 1;
  }
  .summary-card .sc-label {
    font-size: 10px; letter-spacing: .12em; color: var(--dim);
    text-transform: uppercase; margin-top: 5px;
  }

  #detections-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; }

  .det-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    overflow: hidden; cursor: pointer; transition: border-color .2s, transform .15s; position: relative;
  }
  .det-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .det-card.rank1 { border-color: var(--accent2); box-shadow: var(--glow2); }
  .det-card.rank1 .det-rank { background: var(--accent2); color: #fff; }

  .det-img-wrap { position: relative; aspect-ratio: 1; background: #000; overflow: hidden; }
  .det-img-wrap img {
    width: 100%; height: 100%; object-fit: cover; display: block;
    filter: brightness(.9) contrast(1.1); image-rendering: pixelated;
  }
  .det-bbox {
    position: absolute; border: 2px solid var(--accent2);
    box-shadow: 0 0 8px rgba(255,79,0,.6); pointer-events: none;
  }
  .rank1 .det-bbox { border-color: #ff0; box-shadow: 0 0 10px rgba(255,255,0,.7); }

  .det-rank {
    position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,.75);
    border: 1px solid var(--border); border-radius: 3px; font-family: var(--mono);
    font-size: 10px; padding: 2px 6px; color: var(--dim); backdrop-filter: blur(4px);
  }

  .det-info { padding: 10px 12px; }
  .det-chip-name {
    font-family: var(--mono); font-size: 10px; color: var(--textlo); margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .det-conf-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .det-conf-label { font-size: 10px; color: var(--dim); letter-spacing: .08em; text-transform: uppercase; }
  .det-conf-val   { font-family: var(--mono); font-size: 14px; color: var(--accent); margin-left: auto; }
  .rank1 .det-conf-val { color: var(--accent2); }

  .det-bar-outer { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .det-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; }
  .rank1 .det-bar-inner { background: var(--accent2); }

  .det-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 8px; margin-top: 8px; }
  .det-metric { font-size: 9px; color: var(--textlo); display: flex; justify-content: space-between; }
  .det-metric span { color: var(--text); font-family: var(--mono); }

  #modal-overlay {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(4,6,13,.88); backdrop-filter: blur(6px);
    align-items: center; justify-content: center;
  }
  #modal-overlay.open { display: flex; }
  #modal-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    max-width: 640px; width: 92%; padding: 28px; position: relative;
  }
  #modal-close {
    position: absolute; top: 14px; right: 16px; background: none; border: none;
    color: var(--dim); font-size: 20px; cursor: pointer;
  }
  #modal-close:hover { color: var(--accent2); }
  #modal-img {
    width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px;
    image-rendering: pixelated; margin-bottom: 18px;
  }
  #modal-title { font-family: var(--mono); font-size: 14px; color: var(--accent); margin-bottom: 16px; }
  .modal-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .mm-row { display: flex; flex-direction: column; gap: 4px; }
  .mm-label { font-size: 10px; letter-spacing: .12em; color: var(--dim); text-transform: uppercase; }
  .mm-bar-outer { height: 4px; background: var(--border); border-radius: 2px; }
  .mm-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; }
  .mm-val { font-family: var(--mono); font-size: 12px; color: var(--text); }

  #export-btn {
    margin-top: 18px; padding: 10px 20px; background: transparent;
    border: 1px solid var(--border); border-radius: 4px; color: var(--dim);
    font-family: var(--mono); font-size: 12px; letter-spacing: .12em; cursor: pointer;
    transition: border-color .2s, color .2s;
  }
  #export-btn:hover { border-color: var(--accent); color: var(--accent); }

  .error-banner {
    background: rgba(255,79,79,.1); border: 1px solid #ff4f4f; border-radius: 4px;
    padding: 12px 16px; font-family: var(--mono); font-size: 12px; color: #ff8a8a; margin-top: 12px;
  }

  @keyframes scanY {
    0%,100% { transform: translateY(0); opacity: 1; }
    50%      { transform: translateY(28px); opacity: .4; }
  }
  .logo-cross { animation: scanY 3s ease-in-out infinite; }
</style>
</head>
<body>
<div class="shell">

  <!-- HEADER -->
  <header>
    <div class="logo-mark"><div class="logo-cross"></div></div>
    <div class="logo-text">
      <h1>XENARCH</h1>
      <p>Planetary Surface Technosignature Detection — Mk19</p>
    </div>
    <div class="header-badge" id="status-badge">
      BACKEND<br>
      <span id="badge-torch">LOADING…</span><br>
      <span id="badge-train"></span>
    </div>
  </header>

  <div class="main-grid">

    <!-- LEFT: Controls -->
    <aside>
      <div class="panel">
        <div class="panel-title">// Input Images</div>

        <div id="dropzone">
          <input type="file" id="file-input" multiple accept=".tif,.tiff,.png,.jpg,.jpeg,.npy" />
          <div class="dz-icon">⊕</div>
          <div class="dz-label">Drop planetary imagery here<br><span>or click to browse</span><br>TIF · PNG · JPG · NPY</div>
        </div>
        <div id="file-list"></div>

        <div class="panel-title" style="margin-top:24px">// Detection Parameters</div>
        <div class="param-grid">
          <div class="param-block">
            <label>Chip Size (px)</label>
            <select id="p-chip">
              <option value="128">128</option>
              <option value="256" selected>256</option>
              <option value="512">512</option>
            </select>
          </div>
          <div class="param-block">
            <label>Chip Overlap</label>
            <select id="p-overlap">
              <option value="0">0%</option>
              <option value="25">25%</option>
              <option value="50" selected>50%</option>
            </select>
          </div>
          <div class="param-block">
            <label>Anomaly %ile</label>
            <input type="number" id="p-pct" value="92" min="50" max="99" step="1" />
          </div>
          <div class="param-block">
            <label>Train Trim %</label>
            <input type="number" id="p-trim" value="8" min="0" max="25" step="1" />
          </div>
          <div class="param-block">
            <label>Epochs</label>
            <input type="number" id="p-epochs" value="20" min="1" max="50" />
          </div>
          <div class="param-block">
            <label>Latent Dim</label>
            <input type="number" id="p-latent" value="56" min="8" max="256" />
          </div>
          <div class="param-block">
            <label>Batch Size</label>
            <input type="number" id="p-batch" value="4" min="1" max="32" />
          </div>
          <div class="param-block">
            <label>Learn Rate</label>
            <input type="number" id="p-lr" value="0.0005" min="0.00001" max="0.01" step="0.00001" />
          </div>
        </div>

        <button id="run-btn" disabled>▶ RUN ANALYSIS</button>
        <div id="error-area"></div>
      </div>
    </aside>

    <!-- RIGHT: Progress + Results -->
    <div class="right-col">

      <!-- PROGRESS -->
      <div class="panel" id="progress-panel">
        <div class="panel-title">// Pipeline Status</div>
        <div class="progress-steps">
          <div class="step-item" id="s1">1 · Chip Extract</div>
          <div class="step-item" id="s2">2 · Robust Train</div>
          <div class="step-item" id="s3">3 · Normalise</div>
          <div class="step-item" id="s4">4 · Rank</div>
          <div class="step-item" id="s5">5 · Package</div>
        </div>
        <div class="prog-bar-outer"><div class="prog-bar-inner" id="prog-bar"></div></div>
        <div id="log-box"></div>
      </div>

      <!-- RESULTS -->
      <div class="panel" id="results-panel">
        <div class="panel-title">// Detection Results</div>
        <div class="summary-row" id="summary-row"></div>
        <div id="detections-grid"></div>
        <button id="export-btn">⬇ Export CSV</button>
      </div>

    </div><!-- /right-col -->
  </div><!-- /main-grid -->
</div><!-- /shell -->

<!-- MODAL -->
<div id="modal-overlay">
  <div id="modal-box">
    <button id="modal-close">✕</button>
    <div id="modal-title"></div>
    <img id="modal-img" src="" alt="chip" />
    <div class="modal-metrics" id="modal-metrics"></div>
  </div>
</div>

<script>
/* ── API base: auto-detect same host ───────────────────────────────────── */
const API = window.location.origin;

/* ── Status badge ──────────────────────────────────────────────────────── */
async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    const badge = document.getElementById('badge-torch');
    badge.textContent = d.torch ? 'VAE-Mk19 (PyTorch)' : 'NumPy robust scorer';
    badge.style.color = d.torch ? 'var(--accent)' : '#ffb347';
    const tb = document.getElementById('badge-train');
    tb.textContent = d.training_images > 0
      ? `BASELINE · ${d.training_images} IMG`
      : 'BASELINE · SELF (training/ empty)';
    tb.style.color = d.training_images > 0 ? 'var(--accent)' : '#ffb347';
  } catch(e) {
    document.getElementById('badge-torch').textContent = 'OFFLINE';
  }
}
checkStatus();

/* ── File handling ─────────────────────────────────────────────────────── */
let selectedFiles = [];
const dz     = document.getElementById('dropzone');
const fi     = document.getElementById('file-input');
const fl     = document.getElementById('file-list');
const runBtn = document.getElementById('run-btn');

fi.addEventListener('change', () => addFiles(fi.files));
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); addFiles(e.dataTransfer.files); });

function addFiles(raw) {
  Array.from(raw).forEach(f => {
    if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
  });
  renderFileList();
}
function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderFileList();
}
function renderFileList() {
  fl.innerHTML = '';
  selectedFiles.forEach(f => {
    const tag = document.createElement('div');
    tag.className = 'file-tag';
    const kb = (f.size / 1024).toFixed(0);
    tag.innerHTML = `<span>${f.name} <span style="color:var(--textlo)">${kb}KB</span></span>
                     <button onclick="removeFile('${f.name}')">×</button>`;
    fl.appendChild(tag);
  });
  runBtn.disabled = selectedFiles.length === 0;
}

/* ── Run analysis ──────────────────────────────────────────────────────── */
let activeJobId = null;
let pollTimer   = null;
let csvData     = [];

runBtn.addEventListener('click', startAnalysis);

async function startAnalysis() {
  if (!selectedFiles.length) return;
  document.getElementById('error-area').innerHTML = '';

  const config = {
    chip_size:    +document.getElementById('p-chip').value,
    overlap:      +document.getElementById('p-overlap').value / 100,
    percentile:   +document.getElementById('p-pct').value,
    trim_frac:    +document.getElementById('p-trim').value / 100,
    epochs:       +document.getElementById('p-epochs').value,
    latent_dim:   +document.getElementById('p-latent').value,
    batch_size:   +document.getElementById('p-batch').value,
    lr:           +document.getElementById('p-lr').value,
  };

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files[]', f));
  fd.append('config', JSON.stringify(config));

  runBtn.disabled = true;
  showProgress(true);
  resetSteps();
  document.getElementById('results-panel').style.display = 'none';
  clearLog();

  try {
    const r = await fetch(`${API}/api/analyze`, { method:'POST', body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    activeJobId = d.job_id;
    pollTimer = setInterval(pollProgress, 1000);
  } catch(e) {
    showError(e.message);
    runBtn.disabled = false;
  }
}

async function pollProgress() {
  if (!activeJobId) return;
  try {
    const r = await fetch(`${API}/api/progress/${activeJobId}`);
    const d = await r.json();

    setProgress(d.pct, d.step);
    (d.logs || []).forEach(appendLog);

    if (d.error) {
      clearInterval(pollTimer);
      showError(d.error);
      runBtn.disabled = false;
      return;
    }

    if (d.done) {
      clearInterval(pollTimer);
      await loadResults();
      runBtn.disabled = false;
    }
  } catch(e) { /* network blip, retry */ }
}

async function loadResults() {
  const r = await fetch(`${API}/api/results/${activeJobId}`);
  const d = await r.json();
  if (d.error) { showError(d.error); return; }
  csvData = d.csv_rows || [];
  renderSummary(d.summary);
  renderDetections(d.detections);
  document.getElementById('results-panel').style.display = 'block';
}

/* ── UI helpers ────────────────────────────────────────────────────────── */
function showProgress(on) {
  document.getElementById('progress-panel').style.display = on ? 'block' : 'none';
}

function resetSteps() {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
  }
  document.getElementById('prog-bar').style.width = '0%';
}

function setProgress(pct, step) {
  document.getElementById('prog-bar').style.width = pct + '%';
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
    if (i < step) el.classList.add('done');
    else if (i === step) el.classList.add('active');
  }
}

const logBox = document.getElementById('log-box');
const seenLogs = new Set();
function appendLog(entry) {
  const key = entry.t + entry.msg;
  if (seenLogs.has(key)) return;
  seenLogs.add(key);
  const cls = entry.level === 'success' ? 'log-ok' : entry.level === 'error' ? 'log-err' : entry.level === 'warning' ? 'log-warn' : '';
  logBox.innerHTML += `<div class="${cls}">[${entry.t}] ${entry.msg}</div>`;
  logBox.scrollTop = logBox.scrollHeight;
}
function clearLog() { logBox.innerHTML = ''; seenLogs.clear(); }

function showError(msg) {
  document.getElementById('error-area').innerHTML = `<div class="error-banner">ERROR: ${msg}</div>`;
}

/* ── Summary cards ─────────────────────────────────────────────────────── */
function renderSummary(s) {
  const row = document.getElementById('summary-row');
  row.innerHTML = [
    { val: s.total_chips,  label: 'Total Chips' },
    { val: s.n_anomalies,  label: 'Anomalies' },
    { val: s.n_high_conf,  label: 'High Conf' },
    { val: (s.top_conf*100).toFixed(2)+'%', label: 'Top Confidence' },
  ].map(c => `
    <div class="summary-card">
      <div class="sc-val">${c.val}</div>
      <div class="sc-label">${c.label}</div>
    </div>`).join('');
}

/* ── Detection grid ────────────────────────────────────────────────────── */
function renderDetections(dets) {
  const grid = document.getElementById('detections-grid');
  grid.innerHTML = '';
  dets.forEach((d, idx) => {
    const conf = (d.confidence * 100).toFixed(2);
    const card = document.createElement('div');
    card.className = 'det-card' + (idx === 0 ? ' rank1' : '');
    card.onclick = () => openModal(d);

    let bboxHtml = '';
    if (d.featureBbox) {
      const [y1n, x1n, y2n, x2n] = d.featureBbox;
      bboxHtml = `<div class="det-bbox" style="
        top:${(y1n*100).toFixed(1)}%;
        left:${(x1n*100).toFixed(1)}%;
        width:${((x2n-x1n)*100).toFixed(1)}%;
        height:${((y2n-y1n)*100).toFixed(1)}%;
      "></div>`;
    }

    const mkeys = ['mse','latent','contextual','gradient','edge'];
    const mrows = mkeys.map(k =>
      `<div class="det-metric">${k} <span>${d.metrics[k]}</span></div>`
    ).join('');

    card.innerHTML = `
      <div class="det-img-wrap">
        <img src="${d.imgDataURI}" alt="chip" />
        ${bboxHtml}
        <div class="det-rank">#${d.rank}</div>
      </div>
      <div class="det-info">
        <div class="det-chip-name">${d.chipName}</div>
        <div class="det-conf-row">
          <div class="det-conf-label">CONF</div>
          <div class="det-conf-val">${conf}%</div>
        </div>
        <div class="det-bar-outer"><div class="det-bar-inner" style="width:${conf}%"></div></div>
        <div class="det-metrics">${mrows}</div>
      </div>`;
    grid.appendChild(card);
  });
}

/* ── Modal ─────────────────────────────────────────────────────────────── */
const overlay = document.getElementById('modal-overlay');
document.getElementById('modal-close').onclick = () => overlay.classList.remove('open');
overlay.addEventListener('click', e => { if (e.target === overlay) overlay.classList.remove('open'); });

function openModal(d) {
  document.getElementById('modal-title').textContent =
    `RANK #${d.rank}  ·  ${d.chipName}  ·  ${(d.confidence*100).toFixed(2)}% confidence`;
  document.getElementById('modal-img').src = d.imgDataURI;

  const mm = document.getElementById('modal-metrics');
  mm.innerHTML = Object.entries(d.metrics).map(([k, v]) => `
    <div class="mm-row">
      <div class="mm-label">${k}</div>
      <div class="mm-bar-outer"><div class="mm-bar-inner" style="width:${(v*100).toFixed(0)}%"></div></div>
      <div class="mm-val">${v}</div>
    </div>`).join('');

  overlay.classList.add('open');
}

/* ── CSV export ────────────────────────────────────────────────────────── */
document.getElementById('export-btn').addEventListener('click', () => {
  if (!csvData.length) return;
  const cols = ['chipName','confidence','score','source',
                'mse','latent','contextual','gradient','edge'];
  const header = cols.join(',');
  const rows = csvData.map(r => [
    r.chipName, r.confidence, r.score, r.source,
    r.metrics.mse, r.metrics.latent, r.metrics.contextual,
    r.metrics.gradient, r.metrics.edge
  ].join(','));
  const blob = new Blob([header+'\n'+rows.join('\n')], {type:'text/csv'});
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: 'xenarch_mk19_results.csv'
  });
  a.click();
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(FRONTEND_HTML)


@app.route("/api/status")
def api_status():
    train_imgs = list_training_images(TRAINING_DIR)
    return jsonify({"status": "ok", "version": "mk19",
                    "torch": HAS_TORCH, "rasterio": HAS_RASTERIO,
                    "training_dir": str(TRAINING_DIR),
                    "training_images": len(train_imgs),
                    "baseline": "training folder" if len(train_imgs) else "self-supervised",
                    "combined_weights": COMBINED_WEIGHTS,
                    "active_jobs": sum(1 for p in PROGRESS.values() if not p.get("done")),
                    "retained_jobs": len(PROGRESS)})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "files[]" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    config = {}
    if "config" in request.form:
        try:
            config = json.loads(request.form["config"])
        except Exception:
            pass
    tmp_dir  = tempfile.mkdtemp(prefix="xenarch_")
    uploaded = []
    for f in request.files.getlist("files[]"):
        if not f.filename:
            continue
        dest = os.path.join(tmp_dir, os.path.basename(f.filename))
        f.save(dest)
        uploaded.append(dest)
    if not uploaded:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": "No files uploaded"}), 400

    job_id = f"job_{int(time.time()*1000)}"
    # Registered so _evict_jobs can delete the chip scratch dir with the job.
    JOB_DIRS[job_id] = tmp_dir
    threading.Thread(target=run_analysis,
                     args=(job_id, uploaded, config, tmp_dir), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    p = PROGRESS.get(job_id)
    if p is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"step": p["step"], "pct": p["pct"],
                    "logs": p["logs"][-50:], "done": p["done"], "error": p["error"]})


@app.route("/api/results/<job_id>")
def api_results(job_id):
    r = RESULTS.get(job_id)
    if r is None:
        p = PROGRESS.get(job_id, {})
        if p.get("error"):
            return jsonify({"error": p["error"]}), 500
        return jsonify({"error": "results not ready"}), 202
    return jsonify(r)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    host  = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("=" * 60)
    logger.info("XENARCH Mk19 — Production Web Edition")
    logger.info(f"  PyTorch  : {'enabled' if HAS_TORCH else 'DISABLED (numpy fallback)'}")
    logger.info(f"  Rasterio : {'enabled' if HAS_RASTERIO else 'disabled (Pillow fallback)'}")
    _n_train = len(list_training_images(TRAINING_DIR))
    _train_note = (f"{_n_train} image(s) found" if _n_train
                   else "EMPTY — self-supervised fallback")
    logger.info(f"  Training : {TRAINING_DIR} ({_train_note})")
    logger.info(f"  CORS     : {CORS_ORIGINS}")
    logger.info(f"  Serving  : http://{host}:{port}")
    logger.info("=" * 60)
    app.run(host=host, port=port, debug=debug, threaded=True)
