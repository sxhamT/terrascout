# eval/synthetic_terrain.py
# Synthetic terrain + LiDAR point cloud generator for offline evaluation.
# Runs entirely in numpy — no Isaac Sim, no ROS2, no GPU required.
#
# Generates N terrain variants with known ground-truth zone labels.
# Each variant is a 20x20 m area sampled into a dense point cloud,
# then classified by lidar_classifier.py to compute precision/recall.
#
# Usage (WSL2 or Windows with any Python 3.11 + numpy/scipy):
#   python eval/synthetic_terrain.py           # 50 variants, results to eval/results/
#   python eval/synthetic_terrain.py --n 10    # quick smoke test
#   python eval/synthetic_terrain.py --seed 0  # reproducible

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.lidar_classifier import LidarClassifier
from terrain.zone_manager import ZoneManager, ZoneStatus, SAFE_THRESHOLD

# ── Constants ──────────────────────────────────────────────────────────────────
TERRAIN_SIZE   = 10.0    # half-extent [m]
CELL_SIZE      = 1.0     # grid resolution [m]
SCAN_ALTITUDE  = 8.0     # synthetic LiDAR ray origin height [m]
POINTS_PER_M2  = 15      # synthetic LiDAR density (RTX LiDAR ~10-20 pts/m² at 8m)
N_VARIANTS     = 50
RESULTS_DIR    = Path(__file__).parent / "results"


# ── Terrain variant generator ─────────────────────────────────────────────────

class TerrainVariant:
    """
    One synthetic terrain instance with known ground-truth zone labels.

    Ground truth: a cell is labelled SAFE if and only if:
      - slope   < 10°
      - no obstacle within 0.6 m of cell centre
      - no high-roughness surface (RMS residual < 0.15 m)
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.terrain_size = TERRAIN_SIZE
        self.cell_size = CELL_SIZE
        n = int(2 * TERRAIN_SIZE / CELL_SIZE)
        self.n = n

        # Ground-truth safety label per cell (bool array)
        self._gt_safe = np.ones((n, n), dtype=bool)

        # List of terrain features: each is a dict with 'type' and parameters
        self._features: List[dict] = []

        self._generate()

    def _generate(self):
        """Procedurally place terrain features and compute ground truth."""
        # Random number of boulders
        n_boulders = int(self.rng.integers(4, 12))
        for _ in range(n_boulders):
            angle = self.rng.uniform(0, 2 * np.pi)
            r     = self.rng.uniform(1.5, 9.5)
            cx    = r * np.cos(angle)
            cy    = r * np.sin(angle)
            radius = self.rng.uniform(0.25, 0.90)
            self._features.append({"type": "boulder", "cx": cx, "cy": cy, "r": radius})
            # Mark cells within (radius + clearance_threshold) as unsafe
            self._mark_unsafe_radius(cx, cy, radius + 0.60)

        # Random number of slope ramps
        n_ramps = int(self.rng.integers(1, 4))
        for _ in range(n_ramps):
            angle = self.rng.uniform(0, 2 * np.pi)
            r     = self.rng.uniform(2.0, 8.0)
            cx    = r * np.cos(angle)
            cy    = r * np.sin(angle)
            slope_deg = self.rng.uniform(12, 30)
            ramp_len  = self.rng.uniform(2.0, 5.0)
            ramp_dir  = self.rng.uniform(0, 2 * np.pi)
            self._features.append({
                "type": "ramp", "cx": cx, "cy": cy,
                "slope_deg": slope_deg, "length": ramp_len, "dir": ramp_dir,
            })
            self._mark_unsafe_ramp(cx, cy, slope_deg, ramp_len, ramp_dir)

        # Guaranteed safe pad (radius 1.5 m) for at least one safe zone
        px = self.rng.uniform(-5.0, 5.0)
        py = self.rng.uniform(-5.0, 5.0)
        self._features.append({"type": "pad", "cx": px, "cy": py, "r": 1.5})
        # Pad overrides any previous unsafe marking within its radius
        self._mark_safe_radius(px, py, 1.2)

    # ── Ground-truth marking ──────────────────────────────────────────────────

    def _cell_centres(self):
        """Return (N²,2) array of cell centre (x,y) coordinates."""
        origin = -self.terrain_size
        cols = np.arange(self.n) * self.cell_size + origin + self.cell_size / 2
        rows = np.arange(self.n) * self.cell_size + origin + self.cell_size / 2
        cx, cy = np.meshgrid(cols, rows)
        return cx, cy

    def _mark_unsafe_radius(self, cx, cy, r):
        cxg, cyg = self._cell_centres()
        dist = np.sqrt((cxg - cx) ** 2 + (cyg - cy) ** 2)
        self._gt_safe[dist < r] = False

    def _mark_safe_radius(self, cx, cy, r):
        cxg, cyg = self._cell_centres()
        dist = np.sqrt((cxg - cx) ** 2 + (cyg - cy) ** 2)
        self._gt_safe[dist < r] = True

    def _mark_unsafe_ramp(self, cx, cy, slope_deg, length, ramp_dir):
        """Mark cells on the ramp face as unsafe."""
        cxg, cyg = self._cell_centres()
        # Project cell centres onto ramp axis
        dx = cxg - cx
        dy = cyg - cy
        along = dx * np.cos(ramp_dir) + dy * np.sin(ramp_dir)
        across = np.abs(-dx * np.sin(ramp_dir) + dy * np.cos(ramp_dir))
        on_ramp = (along > -0.5) & (along < length) & (across < 1.5)
        self._gt_safe[on_ramp] = False

    # ── Point cloud synthesis ─────────────────────────────────────────────────

    def generate_point_cloud(self) -> np.ndarray:
        """
        Simulate a top-down LiDAR scan from SCAN_ALTITUDE.
        Returns (N, 3) XYZ array in world frame.
        """
        origin = -self.terrain_size
        # Dense grid of ray origins on the ground plane
        n_pts_side = int(2 * self.terrain_size * np.sqrt(POINTS_PER_M2))
        xs = self.rng.uniform(-self.terrain_size, self.terrain_size, n_pts_side ** 2)
        ys = self.rng.uniform(-self.terrain_size, self.terrain_size, n_pts_side ** 2)

        # Compute ground height at each sample point
        zs = self._height_at(xs, ys)

        # Add small sensor noise (0.02 m std dev)
        zs += self.rng.normal(0, 0.02, size=len(zs))

        pts = np.column_stack([xs, ys, zs]).astype(np.float32)
        return pts

    def _height_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Compute terrain height at (xs, ys) from all features."""
        zs = np.zeros(len(xs))
        for feat in self._features:
            if feat["type"] == "boulder":
                dist = np.sqrt((xs - feat["cx"]) ** 2 + (ys - feat["cy"]) ** 2)
                r = feat["r"]
                cap = np.clip(1.0 - (dist / r) ** 2, 0, None)
                zs += r * np.sqrt(np.clip(cap, 0, None))
            elif feat["type"] == "ramp":
                dx = xs - feat["cx"]
                dy = ys - feat["cy"]
                along  = dx * np.cos(feat["dir"]) + dy * np.sin(feat["dir"])
                across = np.abs(-dx * np.sin(feat["dir"]) + dy * np.cos(feat["dir"]))
                on_ramp = (along > 0) & (along < feat["length"]) & (across < 1.5)
                slope_m = np.tan(np.radians(feat["slope_deg"]))
                zs += np.where(on_ramp, along * slope_m, 0.0)
        return zs

    @property
    def ground_truth_safe(self) -> np.ndarray:
        """(n, n) bool array — True if cell is genuinely safe to land on."""
        return self._gt_safe.copy()

    @property
    def n_safe_cells(self) -> int:
        return int(self._gt_safe.sum())
