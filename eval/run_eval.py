# eval/run_eval.py
# Offline evaluation harness for TerraScout LiDAR hazard classifier.
# Runs entirely in numpy/scipy — no Isaac Sim, no GPU, no ROS2.
#
# Generates N terrain variants with ground-truth zone labels,
# runs lidar_classifier.py on the synthetic point clouds,
# and produces precision/recall/F1 metrics + per-variant results.
#
# Where:  D:\project\terrascout  (Windows)  OR  /mnt/d/project/terrascout (WSL2)
# Admin:  No
# Run:    python eval/run_eval.py [--n 50] [--seed 42] [--out eval/results]
#
# Output:
#   eval/results/summary.json       machine-readable metrics
#   eval/results/per_variant.csv    per-variant breakdown
#   eval/results/summary.txt        human-readable table for report

import sys
import argparse
import json
import csv
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.synthetic_terrain import TerrainVariant, TERRAIN_SIZE, CELL_SIZE, N_VARIANTS
from controller.terrain_classifier import TerrainClassifier
from terrain.zone_manager import ZoneManager, ZoneStatus, SAFE_THRESHOLD


def run_variant(variant: TerrainVariant) -> dict:
    """
    Run the full classify pipeline on one terrain variant.
    Uses TerrainClassifier (Kalman+Bayesian) — same stack as the sim.
    Returns a dict of metrics for that variant.
    """
    # Build fresh classifier and zone manager — same params as sim
    classifier = TerrainClassifier(terrain_size=TERRAIN_SIZE, cell_size=CELL_SIZE)
    zone_mgr   = ZoneManager(terrain_size=TERRAIN_SIZE, cell_size=CELL_SIZE)

    # Generate synthetic point cloud and classify via full Kalman+Bayesian pipeline
    pts = variant.generate_point_cloud()
    classifier.add_scan(pts)

    # Push p_safe + Kalman variance into zone manager (same as ooda_backend DESCEND_SCAN)
    results = classifier.get_all_results()
    for (row, col), res in results.items():
        wx, wy = classifier.cell_to_world_centre(row, col)
        zone_mgr.update_cell(wx, wy, res["p_safe"], res["variance"])

    # Build predicted safe grid (n x n bool)
    n = variant.n
    pred_safe = np.zeros((n, n), dtype=bool)
    for row in range(n):
        for col in range(n):
            if zone_mgr._status[row, col] == ZoneStatus.SAFE:
                pred_safe[row, col] = True

    gt_safe = variant.ground_truth_safe

    # Only evaluate cells that were actually observed (have a score)
    observed = ~np.isnan(zone_mgr._scores)

    TP = int(np.sum( pred_safe &  gt_safe & observed))
    FP = int(np.sum( pred_safe & ~gt_safe & observed))
    FN = int(np.sum(~pred_safe &  gt_safe & observed))
    TN = int(np.sum(~pred_safe & ~gt_safe & observed))

    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    recall    = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))

    # False positive rate at terrain edges
    edge_mask = _edge_mask(gt_safe)
    edge_obs  = edge_mask & observed
    fp_edge   = int(np.sum(pred_safe & ~gt_safe & edge_obs))
    edge_total = int(edge_obs.sum())
    fpr_edge  = fp_edge / edge_total if edge_total > 0 else float("nan")

    # Best zone accuracy — same cluster logic as sim
    best = zone_mgr.best_landing_zone(min_radius_m=0.7, margin=2.0)
    if best is not None:
        bx, by, bscore = best
        bc = zone_mgr.world_to_cell(bx, by)
        zone_correct = bool(gt_safe[bc[0], bc[1]]) if bc else False
    else:
        zone_correct = False

    coverage = zone_mgr.coverage_fraction()

    return {
        "seed":         variant.seed,
        "n_gt_safe":    variant.n_safe_cells,
        "n_pred_safe":  int(pred_safe.sum()),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "f1":           round(f1, 4),
        "fp_rate_edge": round(fpr_edge, 4),
        "zone_correct": zone_correct,
        "coverage":     round(coverage, 4),
        "n_pts":        len(pts),
    }


def _edge_mask(gt: np.ndarray) -> np.ndarray:
    """True for cells adjacent (4-connected) to a cell with a different GT label."""
    from scipy.ndimage import binary_dilation
    n = gt.shape[0]
    edge = np.zeros_like(gt)
    struct = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)
    dilated_safe   = binary_dilation(gt,  structure=struct)
    dilated_unsafe = binary_dilation(~gt, structure=struct)
    edge = dilated_safe & dilated_unsafe
    return edge


def aggregate(results: list) -> dict:
    """Compute aggregate statistics across all variants."""
    def nanmean(key):
        vals = [r[key] for r in results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    def nanstd(key):
        vals = [r[key] for r in results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return round(float(np.std(vals)), 4) if vals else float("nan")

    zone_acc = sum(1 for r in results if r["zone_correct"]) / len(results)

    return {
        "n_variants":         len(results),
        "precision_mean":     nanmean("precision"),
        "precision_std":      nanstd("precision"),
        "recall_mean":        nanmean("recall"),
        "recall_std":         nanstd("recall"),
        "f1_mean":            nanmean("f1"),
        "f1_std":             nanstd("f1"),
        "fp_rate_edge_mean":  nanmean("fp_rate_edge"),
        "fp_rate_edge_std":   nanstd("fp_rate_edge"),
        "zone_selection_acc": round(zone_acc, 4),
        "coverage_mean":      nanmean("coverage"),
    }


def print_summary(agg: dict, elapsed: float):
    lines = [
        "",
        "=" * 58,
        "  TerraScout LiDAR Classifier Evaluation",
        "=" * 58,
        f"  Variants evaluated : {agg['n_variants']}",
        f"  Elapsed time       : {elapsed:.1f} s",
        "-" * 58,
        f"  Precision          : {agg['precision_mean']:.3f}  ± {agg['precision_std']:.3f}",
        f"  Recall             : {agg['recall_mean']:.3f}  ± {agg['recall_std']:.3f}",
        f"  F1 score           : {agg['f1_mean']:.3f}  ± {agg['f1_std']:.3f}",
        f"  FP rate (edges)    : {agg['fp_rate_edge_mean']:.3f}  ± {agg['fp_rate_edge_std']:.3f}",
        f"  Zone select acc    : {agg['zone_selection_acc']:.3f}",
        f"  Mean coverage      : {agg['coverage_mean']:.3f}",
        "=" * 58,
        "",
    ]
    print("\n".join(lines))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TerraScout offline evaluation")
    parser.add_argument("--n",    type=int, default=N_VARIANTS, help="Number of terrain variants")
    parser.add_argument("--seed", type=int, default=0,          help="Base random seed")
    parser.add_argument("--out",  type=str, default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Eval] Running {args.n} terrain variants (base seed={args.seed})...")
    t0 = time.time()

    per_variant = []
    for i in range(args.n):
        seed = args.seed + i
        variant = TerrainVariant(seed=seed)
        result  = run_variant(variant)
        per_variant.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.n}] seed={seed}  "
                  f"P={result['precision']:.3f}  R={result['recall']:.3f}  "
                  f"F1={result['f1']:.3f}  zone={'OK' if result['zone_correct'] else 'MISS'}")

    elapsed = time.time() - t0
    agg = aggregate(per_variant)
    summary_text = print_summary(agg, elapsed)

    # Write summary.json
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"aggregate": agg, "per_variant": per_variant}, f, indent=2)

    # Write per_variant.csv
    if per_variant:
        with open(out_dir / "per_variant.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_variant[0].keys())
            writer.writeheader()
            writer.writerows(per_variant)

    # Write summary.txt
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    print(f"[Eval] Results written to {out_dir}/")
    return agg


if __name__ == "__main__":
    main()
