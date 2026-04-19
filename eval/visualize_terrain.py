# eval/visualize_terrain.py
# Quick visual check of what the synthetic terrain variants look like.
# Shows heightmap + ground-truth safe zones side by side.
# Run from WSL2 or Windows: python eval/visualize_terrain.py
#
# No Isaac Sim needed — pure numpy + matplotlib.

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.synthetic_terrain import TerrainVariant, TERRAIN_SIZE, CELL_SIZE


def plot_variant(seed: int, ax_height, ax_safe):
    """Plot heightmap and ground-truth safe zones for one terrain variant."""
    tv = TerrainVariant(seed)
    pts = tv.generate_point_cloud()

    # Bin point cloud into a heightmap grid for display
    n = tv.n
    origin = -TERRAIN_SIZE
    cols = ((pts[:, 0] - origin) / CELL_SIZE).astype(int).clip(0, n - 1)
    rows = ((pts[:, 1] - origin) / CELL_SIZE).astype(int).clip(0, n - 1)
    heightmap = np.full((n, n), np.nan)
    for r, c, z in zip(rows, cols, pts[:, 2]):
        if np.isnan(heightmap[r, c]) or z > heightmap[r, c]:
            heightmap[r, c] = z
    heightmap = np.nan_to_num(heightmap, nan=0.0)

    extent = [-TERRAIN_SIZE, TERRAIN_SIZE, -TERRAIN_SIZE, TERRAIN_SIZE]

    # Height map
    im = ax_height.imshow(
        heightmap, origin="lower", extent=extent,
        cmap="terrain", interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax_height, label="height (m)", fraction=0.046)
    ax_height.set_title(f"Variant {seed} — heightmap")
    ax_height.set_xlabel("x (m)")
    ax_height.set_ylabel("y (m)")

    # Ground truth safe zones
    safe = tv.ground_truth_safe.astype(float)
    ax_safe.imshow(
        safe, origin="lower", extent=extent,
        cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest",
    )
    ax_safe.set_title(f"Variant {seed} — ground truth (green=SAFE)")
    ax_safe.set_xlabel("x (m)")
    ax_safe.set_ylabel("y (m)")


def main():
    # Show 6 different variants in a 3×2 grid (3 seeds × 2 plots each)
    seeds = [0, 7, 13, 21, 37, 42]
    fig, axes = plt.subplots(len(seeds), 2, figsize=(10, 4 * len(seeds)))
    fig.suptitle(
        "TerraScout — synthetic terrain variants\n"
        "(left: LiDAR heightmap  |  right: ground-truth safe zones)",
        fontsize=13,
    )

    for i, seed in enumerate(seeds):
        plot_variant(seed, axes[i, 0], axes[i, 1])

    plt.tight_layout()
    out = Path(__file__).parent / "results" / "terrain_variants.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[visualize_terrain] saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
