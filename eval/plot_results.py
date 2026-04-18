# eval/plot_results.py
# Generate all report figures from eval/results/ JSON files.
# Produces publication-ready plots matching ACM BuildSyS sigconf style:
#   - Single column: 3.33 in wide
#   - Double column: 6.5 in wide
#
# Where:  D:\project\terrascout  (any terminal with matplotlib)
# Admin:  No
# Run:    python eval/plot_results.py
# Output: eval/figures/*.pdf  (import directly into LaTeX)

import sys
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[plot_results] matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ACM sigconf single-column width
COL_W = 3.33

# Colour palette — accessible, prints well in greyscale
C_MPC  = "#1a6faf"   # blue
C_PID  = "#c0392b"   # red
C_SAFE = "#27ae60"   # green
C_UNSAFE = "#c0392b"
C_GRAY = "#888888"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.labelsize":    8,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "lines.linewidth":   1.2,
    "axes.linewidth":    0.6,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
})


# ── 1. Precision / Recall / F1 distribution ───────────────────────────────────

def plot_prf_distribution(data: dict):
    variants = data["per_variant"]
    precisions = [v["precision"] for v in variants if not np.isnan(v["precision"])]
    recalls    = [v["recall"]    for v in variants if not np.isnan(v["recall"])]
    f1s        = [v["f1"]        for v in variants if not np.isnan(v["f1"])]

    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.2))
    positions = [1, 2, 3]
    bdata = [precisions, recalls, f1s]
    labels = ["Precision", "Recall", "F1"]
    colors = ["#2980b9", "#27ae60", "#8e44ad"]

    bp = ax.boxplot(bdata, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classifier performance across terrain variants")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    agg = data["aggregate"]
    caption = (f"n={agg['n_variants']}   "
               f"P={agg['precision_mean']:.3f}±{agg['precision_std']:.3f}   "
               f"R={agg['recall_mean']:.3f}±{agg['recall_std']:.3f}   "
               f"F1={agg['f1_mean']:.3f}±{agg['f1_std']:.3f}")
    ax.text(0.5, -0.22, caption, transform=ax.transAxes,
            ha="center", fontsize=6, color=C_GRAY)

    out = FIGURES_DIR / "fig_prf_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] Saved {out.name}")


# ── 2. Edge FPR vs zone selection accuracy ────────────────────────────────────

def plot_edge_fpr(data: dict):
    variants = data["per_variant"]
    fpr = [v["fp_rate_edge"] for v in variants if not np.isnan(v["fp_rate_edge"])]
    zone_ok = [int(v["zone_correct"]) for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 1.6, 2.0))

    # FPR histogram
    ax = axes[0]
    ax.hist(fpr, bins=15, color=C_MPC, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.mean(fpr), color=C_PID, linewidth=1.2, linestyle="--",
               label=f"Mean {np.mean(fpr):.3f}")
    ax.set_xlabel("False positive rate (edge cells)")
    ax.set_ylabel("Count")
    ax.set_title("Edge FPR distribution")
    ax.legend()

    # Zone selection accuracy bar
    ax = axes[1]
    correct = sum(zone_ok)
    incorrect = len(zone_ok) - correct
    bars = ax.bar(["Correct", "Incorrect"], [correct, incorrect],
                  color=[C_SAFE, C_UNSAFE], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Variants")
    ax.set_title(f"Zone selection accuracy\n{correct}/{len(zone_ok)} = "
                 f"{correct/len(zone_ok):.1%}")
    for bar, val in zip(bars, [correct, incorrect]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=7)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_edge_fpr_zone_acc.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] Saved {out.name}")


# ── 3. MPC vs PID descent comparison ─────────────────────────────────────────

def plot_mpc_vs_pid(bench: dict):
    comp = bench["comparison"]
    mpc_raw = bench["mpc_raw"]
    pid_raw = bench["pid_raw"]

    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2.0, 2.2))

    # a. Final lateral error — box plot
    ax = axes[0]
    mpc_lat = [r["lateral_error_m"] for r in mpc_raw]
    pid_lat = [r["lateral_error_m"] for r in pid_raw]
    bp = ax.boxplot([mpc_lat, pid_lat], positions=[1, 2], widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5))
    bp["boxes"][0].set_facecolor(C_MPC)
    bp["boxes"][1].set_facecolor(C_PID)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MPC", "PID"])
    ax.set_ylabel("Error [m]")
    ax.set_title("Final lateral error")
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    # b. Energy proxy
    ax = axes[1]
    mpc_e = [r["energy_proxy"] for r in mpc_raw]
    pid_e = [r["energy_proxy"] for r in pid_raw]
    bp = ax.boxplot([mpc_e, pid_e], positions=[1, 2], widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5))
    bp["boxes"][0].set_facecolor(C_MPC)
    bp["boxes"][1].set_facecolor(C_PID)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MPC", "PID"])
    ax.set_ylabel("Σ||u||²·dt")
    ax.set_title("Energy proxy")
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    # c. Descent speed violation rate
    ax = axes[2]
    mpc_v = [r["vz_violations"] for r in mpc_raw]
    pid_v = [r["vz_violations"] for r in pid_raw]
    bp = ax.boxplot([mpc_v, pid_v], positions=[1, 2], widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5))
    bp["boxes"][0].set_facecolor(C_MPC)
    bp["boxes"][1].set_facecolor(C_PID)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MPC", "PID"])
    ax.set_ylabel("Fraction of steps")
    ax.set_title("Descent speed violations")
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    mpc_patch = mpatches.Patch(color=C_MPC, label="MPC")
    pid_patch = mpatches.Patch(color=C_PID, label="PID")
    fig.legend(handles=[mpc_patch, pid_patch], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    out = FIGURES_DIR / "fig_mpc_vs_pid.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] Saved {out.name}")


# ── 4. Coverage vs F1 scatter ─────────────────────────────────────────────────

def plot_coverage_vs_f1(data: dict):
    variants = data["per_variant"]
    cov = [v["coverage"] for v in variants]
    f1  = [v["f1"]       for v in variants if not np.isnan(v["f1"])]
    cov_valid = [v["coverage"] for v in variants if not np.isnan(v["f1"])]

    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.2))
    ax.scatter(cov_valid, f1, s=12, alpha=0.7, color=C_MPC, edgecolors="none")
    # Trend line
    z = np.polyfit(cov_valid, f1, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(cov_valid), max(cov_valid), 100)
    ax.plot(xs, p(xs), color=C_PID, linewidth=1.0, linestyle="--", label="Linear fit")
    ax.set_xlabel("Coverage fraction")
    ax.set_ylabel("F1 score")
    ax.set_title("Classification quality vs scan coverage")
    ax.legend()
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    out = FIGURES_DIR / "fig_coverage_vs_f1.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] Saved {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    classifier_path = RESULTS_DIR / "summary.json"
    benchmark_path  = RESULTS_DIR / "mpc_vs_pid.json"

    if classifier_path.exists():
        with open(classifier_path) as f:
            data = json.load(f)
        plot_prf_distribution(data)
        plot_edge_fpr(data)
        plot_coverage_vs_f1(data)
        print(f"[plot] Classifier figures written to {FIGURES_DIR}/")
    else:
        print(f"[plot] {classifier_path} not found. Run eval/run_eval.py first.")

    if benchmark_path.exists():
        with open(benchmark_path) as f:
            bench = json.load(f)
        plot_mpc_vs_pid(bench)
        print(f"[plot] Benchmark figure written to {FIGURES_DIR}/")
    else:
        print(f"[plot] {benchmark_path} not found. Run eval/mpc_vs_pid_benchmark.py first.")


if __name__ == "__main__":
    main()
