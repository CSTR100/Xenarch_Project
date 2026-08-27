"""
Xenarch iterative evaluation harness.

Requires labeled evaluation data under `eval/` (see "Expected inputs" below).
No evaluation data ships with this repository — supply your own.

Usage:
    # Run iteration 1 with default config
    python eval_harness.py --iteration 1

    # Run with custom config
    python eval_harness.py --iteration 2 --config my_config.json

    # List previous iterations
    python eval_harness.py --status

Config JSON keys (all optional; defaults come from xenarch_core):
    chip_size, overlap, percentile, epochs, latent_dim, batch_size, lr,
    warmup_epochs, trim_frac, max_train_chips, training_dir,
    engine:             "mk19" (default) | "mk17" (legacy, comparison only)
    combined_weights:   {mse, latent, contextual, gradient, edge}

Stopping criteria (configurable):
    separation_threshold: float  (default 0.30)
    overfit_patience:     int    (default 2)
    max_iterations:       int    (default 6)

Expected inputs
---------------
eval/regions.json
    {"regions": [{"region_id":    "apollo_11",
                  "split":        "tuning" | "held-out",
                  "anomaly_type": "lander",          # free-form grouping label
                  "chip_size":    256}]}             # optional per-region override

eval/<region_id>/ground_truth.json
    {"image_path": "eval/apollo_11/scene.tif",       # relative to this file's dir
     "low_confidence": false,                        # flag thin/uncertain labeling
     "zones": [{"label": "target",         "bbox": [x1, y1, x2, y2]},
               {"label": "false_positive", "bbox": [x1, y1, x2, y2]},
               {"label": "background",     "bbox": [x1, y1, x2, y2]}]}

A chip is labeled by whether its CENTER falls inside a zone, first match winning,
so list `target` and `false_positive` before any catch-all `background` zone. A
chip of width `chip_size` centred at c spans [c - chip_size/2, c + chip_size/2],
so to label exactly the chips that fully contain a feature spanning [f1, f2], use
a zone from f2 - chip_size/2 to f1 + chip_size/2. A wider zone labels chips that
hold only a clipped corner of the feature as `target`, which drags target
confidence down and understates separation.

`separation` is the mean confidence over `target` chips minus the mean over
`false_positive` chips, so a region needs at least one of each to score.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from xenarch_pipeline import run_pipeline, DEFAULT_CONFIG
from xenarch_core import COMBINED_WEIGHTS, METRIC_KEYS


# ── default harness config ─────────────────────────────────────────────────

HARNESS_DEFAULTS = {
    "separation_threshold": 0.30,
    "overfit_patience":     2,
    "max_iterations":       6,
}


# ── ground truth loading ───────────────────────────────────────────────────

DATA_HELP = (
    "No evaluation data found. This repository ships none — supply your own.\n"
    "Expected layout (see the module docstring for the full schema):\n"
    "    eval/regions.json\n"
    "    eval/<region_id>/ground_truth.json\n"
    "    eval/<region_id>/<scene image>\n"
)


def load_ground_truth(region_id: str) -> Dict:
    path = BASE / "eval" / region_id / "ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ground truth for region {region_id!r}: {path}")
    with open(path) as f:
        return json.load(f)


def load_regions() -> List[Dict]:
    path = BASE / "eval" / "regions.json"
    if not path.exists():
        print(f"\n{DATA_HELP}\nLooked for: {path}\n", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        regions = json.load(f).get("regions", [])
    if not regions:
        print(f"\n{path} lists no regions.\n{DATA_HELP}", file=sys.stderr)
        sys.exit(2)
    return regions


# ── zone matching ──────────────────────────────────────────────────────────

def assign_labels(results: List[Dict], zones: List[Dict]) -> Dict[int, str]:
    """
    Return {chip_id: label} using chip center_x/center_y vs zone bboxes.
    Priority order within zones list: first match wins.
    """
    labels: Dict[int, str] = {}
    for chip in results:
        cx, cy = chip["center_x"], chip["center_y"]
        assigned = "unlabeled"
        for zone in zones:
            x1, y1, x2, y2 = zone["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                assigned = zone["label"]
                break
        labels[chip["chip_id"]] = assigned
    return labels


# ── per-region metrics ─────────────────────────────────────────────────────

def compute_region_metrics(results: List[Dict], ground_truth: Dict) -> Dict:
    labels = assign_labels(results, ground_truth.get("zones", []))

    def group(lbl):
        return [c for c in results if labels.get(c["chip_id"]) == lbl]

    targets = group("target")
    fps     = group("false_positive")
    bgs     = group("background")

    def safe_mean(lst, key):
        return float(np.mean([x[key] for x in lst])) if lst else None

    def safe_mean_dict(lst, keys):
        if not lst:
            return {k: None for k in keys}
        return {k: float(np.mean([x[k] for x in lst])) for k in keys}

    metric_keys = ["mse_norm", "contextual_norm", "gradient_norm", "edge_norm", "latent_norm"]

    t_conf = safe_mean(targets, "confidence")
    f_conf = safe_mean(fps,     "confidence")
    separation = (t_conf - f_conf) if (t_conf is not None and f_conf is not None) else None

    return {
        "n_chips":              len(results),
        "n_target":             len(targets),
        "n_fp":                 len(fps),
        "n_bg":                 len(bgs),
        "target_mean_rank":     safe_mean(targets, "rank"),
        "target_mean_conf":     t_conf,
        "fp_mean_rank":         safe_mean(fps, "rank"),
        "fp_mean_conf":         f_conf,
        "separation":           separation,
        "target_in_top3":       any(c["rank"] <= 3 for c in targets),
        "fp_suppressed":        (f_conf < 0.50) if f_conf is not None else None,
        "low_confidence_region": ground_truth.get("low_confidence", False),
        # per-metric breakdown for diagnosis
        "target_metric_means":  safe_mean_dict(targets, metric_keys),
        "fp_metric_means":      safe_mean_dict(fps, metric_keys),
        # chip-level detail for the report
        "target_chips": [{"chip_id": c["chip_id"], "rank": c["rank"],
                          "confidence": round(c["confidence"], 4),
                          "mse_norm": round(c["mse_norm"], 3),
                          "contextual_norm": round(c["contextual_norm"], 3),
                          "gradient_norm": round(c["gradient_norm"], 3),
                          "edge_norm": round(c["edge_norm"], 3)}
                         for c in targets],
        "fp_chips":     [{"chip_id": c["chip_id"], "rank": c["rank"],
                          "confidence": round(c["confidence"], 4),
                          "mse_norm": round(c["mse_norm"], 3),
                          "contextual_norm": round(c["contextual_norm"], 3),
                          "gradient_norm": round(c["gradient_norm"], 3),
                          "edge_norm": round(c["edge_norm"], 3)}
                         for c in fps],
    }


# ── aggregate metrics ──────────────────────────────────────────────────────

def compute_aggregate(region_results: Dict[str, Dict], split: Optional[str] = None) -> Dict:
    """
    Compute mean + worst-case separation over a split.
    low_confidence regions are included but flagged.
    """
    seps = []
    for rid, data in region_results.items():
        if split and data["split"] != split:
            continue
        m = data["metrics"]
        if m["separation"] is not None:
            seps.append((rid, m["separation"], m.get("low_confidence_region", False)))

    if not seps:
        return {"mean_separation": None, "min_separation": None,
                "worst_region": None, "n_regions": 0}

    vals = [s for _, s, _ in seps]
    worst_idx = int(np.argmin(vals))
    return {
        "mean_separation": float(np.mean(vals)),
        "min_separation":  float(vals[worst_idx]),
        "worst_region":    seps[worst_idx][0],
        "n_regions":       len(seps),
        "low_conf_regions": [rid for rid, _, lc in seps if lc],
    }


# ── stopping criteria ──────────────────────────────────────────────────────

def load_progress_log() -> List[Dict]:
    log_path = BASE / "results" / "progress_log.csv"
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def check_stopping(
    iteration: int,
    tuning_agg: Dict,
    held_agg: Dict,
    threshold: float,
    patience: int,
    max_iter: int,
) -> Tuple[bool, str]:
    """Returns (should_stop, reason)."""
    if iteration >= max_iter:
        return True, f"Reached maximum iterations ({max_iter})."

    t_sep = tuning_agg["mean_separation"]
    h_sep = held_agg["mean_separation"]

    if t_sep is not None and h_sep is not None:
        if t_sep > threshold and h_sep > threshold:
            return True, (f"SUCCESS — both tuning ({t_sep:.3f}) and held-out "
                          f"({h_sep:.3f}) exceed threshold ({threshold}).")

    # overfitting check: look at last `patience` rows in log
    log = load_progress_log()
    tuning_rows  = [r for r in log if r["split"] == "tuning"]
    held_rows    = [r for r in log if r["split"] == "held-out"]

    if len(tuning_rows) >= patience and len(held_rows) >= patience:
        recent_t = [float(r["mean_separation"]) for r in tuning_rows[-patience:]]
        recent_h = [float(r["mean_separation"]) for r in held_rows[-patience:]]
        tuning_improved = recent_t[-1] > recent_t[0]
        held_stagnant   = recent_h[-1] <= recent_h[0]
        if tuning_improved and held_stagnant:
            return True, (
                f"OVERFITTING PLATEAU — tuning improved "
                f"({recent_t[0]:.3f}→{recent_t[-1]:.3f}) but held-out "
                f"stagnated ({recent_h[0]:.3f}→{recent_h[-1]:.3f}) "
                f"over {patience} iterations. Do NOT proceed with current changes."
            )

    return False, ""


# ── progress log ───────────────────────────────────────────────────────────

def append_progress_log(iteration: int, split: str, agg: Dict, config: Dict) -> None:
    log_path = BASE / "results" / "progress_log.csv"
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        fieldnames = ["iteration", "split", "mean_separation", "min_separation",
                      "worst_region", "n_regions", "timestamp",
                      "combined_mse", "combined_latent", "combined_contextual",
                      "combined_gradient", "combined_edge",
                      "conf_score", "conf_contextual", "conf_mse"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        cw = config.get("combined_weights", {})
        kw = config.get("confidence_weights", {})
        writer.writerow({
            "iteration":         iteration,
            "split":             split,
            "mean_separation":   f"{agg['mean_separation']:.4f}" if agg["mean_separation"] is not None else "",
            "min_separation":    f"{agg['min_separation']:.4f}"  if agg["min_separation"]  is not None else "",
            "worst_region":      agg.get("worst_region", ""),
            "n_regions":         agg.get("n_regions", 0),
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "combined_mse":        cw.get("mse", ""),
            "combined_latent":     cw.get("latent", ""),
            "combined_contextual": cw.get("contextual", ""),
            "combined_gradient":   cw.get("gradient", ""),
            "combined_edge":       cw.get("edge", ""),
            "conf_score":          kw.get("score", ""),
            "conf_contextual":     kw.get("contextual", ""),
            "conf_mse":            kw.get("mse", ""),
        })


# ── text report ────────────────────────────────────────────────────────────

def _sep_bar(sep: Optional[float], threshold: float, width: int = 20) -> str:
    if sep is None:
        return "[N/A]"
    filled = int(min(max(sep / 1.0, 0), 1) * width)
    mark   = "✓" if sep >= threshold else " "
    return f"[{'█' * filled}{'░' * (width - filled)}] {sep:+.3f} {mark}"


def print_report(
    iteration: int,
    region_results: Dict[str, Dict],
    tuning_agg: Dict,
    held_agg: Dict,
    stop: bool,
    stop_reason: str,
    config: Dict,
    threshold: float,
) -> None:
    # Mk19 carries its weights in xenarch_core; a config may still override them.
    cw = {**COMBINED_WEIGHTS, **config.get("combined_weights", {})}
    engine = config.get("engine", "mk19")
    print("\n" + "═" * 72)
    print(f"  XENARCH EVAL HARNESS — Iteration {iteration}   "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 72)
    print(f"\nConfig:")
    print(f"  engine={engine}  chip_size={config.get('chip_size',256)}"
          f"  overlap={config.get('overlap',0.5)}"
          f"  percentile={config.get('percentile',92)}"
          f"  epochs={config.get('epochs',20)}  trim_frac={config.get('trim_frac',0.08)}")
    print(f"  combined  : " + "  ".join(
        f"{k}={cw.get(k,0.0):.2f}" for k in METRIC_KEYS))
    print(f"  Stopping threshold: {threshold}")

    for split_label, split_key in [("TUNING SET", "tuning"), ("HELD-OUT SET", "held-out")]:
        agg = tuning_agg if split_key == "tuning" else held_agg
        print(f"\n{'─' * 72}")
        print(f"  {split_label}")
        print(f"  {'Region':<20} {'N':>4} {'Tgt':>4} {'FP':>4}  "
              f"{'Tgt Rank':>8}  {'Tgt Conf':>8}  {'FP Conf':>8}  {'Sep':>7}  Flags")
        print(f"  {'─'*20} {'─'*4} {'─'*4} {'─'*4}  "
              f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
        for rid, data in region_results.items():
            if data["split"] != split_key:
                continue
            m = data["metrics"]
            flags = ""
            if m.get("low_confidence_region"):
                flags += "[low-N] "
            if m["target_in_top3"]:
                flags += "[top3✓] "
            if m["fp_suppressed"]:
                flags += "[fp-ok] "
            sep_str = f"{m['separation']:+.3f}" if m["separation"] is not None else "  N/A "
            trank   = f"{m['target_mean_rank']:.1f}" if m["target_mean_rank"] is not None else "  N/A"
            tconf   = f"{m['target_mean_conf']:.3f}" if m["target_mean_conf"] is not None else "  N/A"
            fconf   = f"{m['fp_mean_conf']:.3f}"     if m["fp_mean_conf"]     is not None else "  N/A"
            print(f"  {rid:<20} {m['n_chips']:>4} {m['n_target']:>4} {m['n_fp']:>4}  "
                  f"{trank:>8}  {tconf:>8}  {fconf:>8}  {sep_str:>7}  {flags}")

        if agg["mean_separation"] is not None:
            print(f"\n  Aggregate  mean_sep={agg['mean_separation']:+.3f}  "
                  f"min_sep={agg['min_separation']:+.3f} ({agg['worst_region']})")
            print(f"  {_sep_bar(agg['mean_separation'], threshold)}")
        else:
            print("  Aggregate: insufficient data")

    # per-metric breakdown for diagnosis
    print(f"\n{'─' * 72}")
    print("  METRIC BREAKDOWN (normalised means per chip class)")
    print(f"  {'Region':<20} {'Class':<8} {'mse':>6} {'ctx':>6} {'grad':>6} {'edge':>6} {'lat':>6}")
    for rid, data in region_results.items():
        m = data["metrics"]
        for cls, md in [("target", m["target_metric_means"]),
                         ("fp",    m["fp_metric_means"])]:
            if all(v is None for v in md.values()):
                continue
            def fmt(v): return f"{v:.3f}" if v is not None else "  N/A"
            print(f"  {rid:<20} {cls:<8} "
                  f"{fmt(md.get('mse_norm')):>6} {fmt(md.get('contextual_norm')):>6} "
                  f"{fmt(md.get('gradient_norm')):>6} {fmt(md.get('edge_norm')):>6} "
                  f"{fmt(md.get('latent_norm')):>6}")

    # anomaly type summary
    print(f"\n{'─' * 72}")
    print("  ANOMALY TYPE PERFORMANCE (tuning + held-out)")
    type_seps: Dict[str, List[float]] = {}
    regions_meta = {r["region_id"]: r for r in load_regions()}
    for rid, data in region_results.items():
        atype = regions_meta[rid]["anomaly_type"]
        sep = data["metrics"]["separation"]
        if sep is not None:
            type_seps.setdefault(atype, []).append(sep)
    for atype, seps_list in type_seps.items():
        avg = np.mean(seps_list)
        flag = "✓" if avg >= threshold else "✗"
        print(f"  {atype:<30} mean_sep={avg:+.3f}  {flag}")

    # stopping
    print(f"\n{'═' * 72}")
    if stop:
        print(f"  STOP: {stop_reason}")
    else:
        print("  → Continue to next iteration.")
    print("═" * 72 + "\n")


# ── main run ───────────────────────────────────────────────────────────────

def run_harness(iteration: int, config: Dict, harness_config: Dict) -> None:
    threshold     = harness_config.get("separation_threshold", 0.30)
    patience      = harness_config.get("overfit_patience",     2)
    max_iter      = harness_config.get("max_iterations",       6)

    regions    = load_regions()
    out_dir    = BASE / "results" / f"iteration_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    region_results: Dict[str, Dict] = {}

    for reg in regions:
        rid        = reg["region_id"]
        split      = reg["split"]
        chip_size  = reg.get("chip_size", config.get("chip_size", 256))

        print(f"\n[{rid}] ({split}) running pipeline…", flush=True)

        gt = load_ground_truth(rid)

        # resolve image path
        img_path = BASE / gt["image_path"]
        if not img_path.exists():
            print(f"  WARNING: image not found at {img_path} — skipping region")
            continue

        # per-region chip_size override
        run_cfg = dict(config)
        run_cfg["chip_size"] = chip_size

        try:
            results = run_pipeline(str(img_path), run_cfg)
        except Exception as e:
            print(f"  ERROR running pipeline: {e}")
            continue

        if not results:
            print(f"  WARNING: no chips returned — skipping region")
            continue

        metrics = compute_region_metrics(results, gt)
        region_results[rid] = {
            "split":   split,
            "metrics": metrics,
        }

        # save per-region chip labels CSV
        region_dir = out_dir / rid
        region_dir.mkdir(exist_ok=True)
        with open(region_dir / "chip_labels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "chip_id", "center_x", "center_y", "label", "rank", "confidence",
                "mse_norm", "latent_norm", "contextual_norm", "gradient_norm", "edge_norm",
            ])
            writer.writeheader()
            label_map = assign_labels(results, gt.get("zones", []))
            for c in sorted(results, key=lambda x: x["rank"]):
                writer.writerow({
                    "chip_id":        c["chip_id"],
                    "center_x":       c["center_x"],
                    "center_y":       c["center_y"],
                    "label":          label_map.get(c["chip_id"], "unlabeled"),
                    "rank":           c["rank"],
                    "confidence":     round(c["confidence"], 4),
                    "mse_norm":       round(c.get("mse_norm", 0), 4),
                    "latent_norm":    round(c.get("latent_norm", 0), 4),
                    "contextual_norm":round(c.get("contextual_norm", 0), 4),
                    "gradient_norm":  round(c.get("gradient_norm", 0), 4),
                    "edge_norm":      round(c.get("edge_norm", 0), 4),
                })

        sep_str = f"{metrics['separation']:+.3f}" if metrics["separation"] is not None else "N/A"
        t_in_top = metrics["target_in_top3"]
        print(f"  chips={metrics['n_chips']}  target={metrics['n_target']} "
              f"fp={metrics['n_fp']}  sep={sep_str}  "
              f"target_in_top3={t_in_top}  "
              f"target_rank={metrics['target_mean_rank']}")

    # aggregates
    tuning_agg = compute_aggregate(region_results, split="tuning")
    held_agg   = compute_aggregate(region_results, split="held-out")

    # stopping
    stop, stop_reason = check_stopping(
        iteration, tuning_agg, held_agg, threshold, patience, max_iter
    )

    # print report
    print_report(iteration, region_results, tuning_agg, held_agg,
                 stop, stop_reason, config, threshold)

    # save full JSON report
    report = {
        "iteration":     iteration,
        "config":        config,
        "harness_config": harness_config,
        "region_results": region_results,
        "tuning_aggregate":   tuning_agg,
        "held_out_aggregate": held_agg,
        "stop":          stop,
        "stop_reason":   stop_reason,
        "timestamp":     datetime.now().isoformat(),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # append to progress log
    for split_key, agg in [("tuning", tuning_agg), ("held-out", held_agg)]:
        if agg["mean_separation"] is not None:
            append_progress_log(iteration, split_key, agg, config)

    print(f"Results saved → {out_dir}")

    if stop:
        sys.exit(0)


# ── status display ─────────────────────────────────────────────────────────

def print_status() -> None:
    log = load_progress_log()
    if not log:
        print("No iterations logged yet. Run with --iteration 1 to start.")
        return
    print(f"\n{'Iter':>4} {'Split':<10} {'Mean Sep':>9} {'Min Sep':>9} {'Worst Region':<22} {'Timestamp'}")
    print("─" * 80)
    for row in log:
        print(f"{row['iteration']:>4} {row['split']:<10} "
              f"{row['mean_separation']:>9} {row['min_separation']:>9} "
              f"{row['worst_region']:<22} {row['timestamp']}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xenarch iterative evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Workflow:
              1.  Populate eval/ with labeled regions (see module docstring)
              2.  python eval_harness.py --iteration 1
              3.  Examine report, adjust config JSON
              4.  python eval_harness.py --iteration 2 --config updated.json
              ...
        """),
    )
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration number (required unless --status)")
    parser.add_argument("--config", default=None,
                        help="Path to config JSON (defaults to Mk19 defaults)")
    parser.add_argument("--separation-threshold", type=float, default=0.30)
    parser.add_argument("--overfit-patience",     type=int,   default=2)
    parser.add_argument("--max-iterations",       type=int,   default=6)
    parser.add_argument("--status", action="store_true",
                        help="Show progress log and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if args.iteration is None:
        parser.error("--iteration is required (or use --status)")

    # load pipeline config
    pipeline_config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            pipeline_config.update(json.load(f))

    harness_config = {
        "separation_threshold": args.separation_threshold,
        "overfit_patience":     args.overfit_patience,
        "max_iterations":       args.max_iterations,
    }

    run_harness(args.iteration, pipeline_config, harness_config)
