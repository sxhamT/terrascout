# scanner/lidar_classifier.py
# LiDAR-based geometric hazard classifier for TerraScout.
# Takes a raw point cloud (N x 3 or N x 4) and scores each terrain cell
# on slope, roughness, and obstacle clearance.
#
# Called by OODABackend.update_graphical_sensor() whenever new LiDAR data arrives.
# Pure numpy — no ROS2, no USD imports.

import numpy as np


# ── Classifier configuration ──────────────────────────────────────────────────

SLOPE_THRESHOLD_DEG   = 10.0   # reject if slope > this
ROUGHNESS_THRESHOLD   = 0.15   # reject if RMS plane residual > this [m]
CLEARANCE_THRESHOLD   = 0.60   # reject if any obstacle within this radius [m]
MIN_POINTS_PER_CELL   = 5      # skip cell if fewer points (not enough data)

W_SLOPE     = 0.40
W_ROUGHNESS = 0.35
W_CLEARANCE = 0.25


class LidarClassifier:
    """
    Classifies terrain cells from an accumulated LiDAR point cloud.

    Usage:
        classifier = LidarClassifier(cell_size=1.0, terrain_size=10.0)
        classifier.add_points(point_cloud_array)   # call each LiDAR frame
        scores = classifier.compute_scores()        # -> dict {(row,col): score}
        classifier.clear()                          # between scan passes
    """

    def __init__(self, cell_size: float = 1.0, terrain_size: float = 10.0):
        self.cell_size = cell_size
        self.terrain_size = terrain_size
        self._origin = -terrain_size
        n = int(2 * terrain_size / cell_size)
        self._n = n
        # Per-cell point lists: dict (row, col) -> list of (x, y, z)
        self._cell_points: dict = {}

    def clear(self):
        """Discard all accumulated points."""
        self._cell_points.clear()

    def add_points(self, points: np.ndarray):
        """
        Accumulate a LiDAR frame into the per-cell buckets.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3) or (N, 4)
            XYZ (+ optional intensity) in the world frame.
            Rows with z < -0.05 are skipped (ground-plane clutter).
        """
        if points.ndim != 2 or points.shape[1] < 3:
            return
        xyz = points[:, :3]
        # Drop points at or below ground level (LiDAR looking down)
        # z is altitude — terrain surface points have small positive z
        xyz = xyz[xyz[:, 2] > -0.05]
        if len(xyz) == 0:
            return

        # Bin into cells
        cols = ((xyz[:, 0] - self._origin) / self.cell_size).astype(int)
        rows = ((xyz[:, 1] - self._origin) / self.cell_size).astype(int)

        valid = (
            (cols >= 0) & (cols < self._n) &
            (rows >= 0) & (rows < self._n)
        )
        for r, c, p in zip(rows[valid], cols[valid], xyz[valid]):
            key = (int(r), int(c))
            if key not in self._cell_points:
                self._cell_points[key] = []
            self._cell_points[key].append(p)

    def compute_scores(self) -> dict:
        """
        Compute safety scores for every cell that has enough points.

        Returns
        -------
        dict mapping (row, col) -> float score in [0, 1]
        """
        scores = {}
        for (row, col), pts_list in self._cell_points.items():
            pts = np.array(pts_list, dtype=np.float32)
            if len(pts) < MIN_POINTS_PER_CELL:
                continue
            s_slope = self._score_slope(pts)
            s_rough = self._score_roughness(pts)
            s_clear = self._score_clearance(pts, row, col)
            score = W_SLOPE * s_slope + W_ROUGHNESS * s_rough + W_CLEARANCE * s_clear
            scores[(row, col)] = float(np.clip(score, 0.0, 1.0))
        return scores

    def cell_to_world_centre(self, row: int, col: int):
        """Convert cell indices to world (x, y) centre coordinates."""
        cx = self._origin + (col + 0.5) * self.cell_size
        cy = self._origin + (row + 0.5) * self.cell_size
        return cx, cy

    # ── Feature extractors ────────────────────────────────────────────────────

    def _score_slope(self, pts: np.ndarray) -> float:
        """
        Estimate surface normal via PCA; score based on tilt from vertical.
        Returns 1.0 for flat, 0.0 for slope >= SLOPE_THRESHOLD_DEG.
        """
        if len(pts) < 3:
            return 0.0
        centred = pts - pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0
        # Smallest singular value → surface normal direction
        normal = Vt[-1]
        # Ensure normal points upward
        if normal[2] < 0:
            normal = -normal
        # Angle between normal and vertical (0,0,1)
        cos_angle = np.clip(normal[2] / (np.linalg.norm(normal) + 1e-9), 0.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        # Linear decay: 0° → score 1.0, threshold → score 0.0
        score = 1.0 - np.clip(angle_deg / SLOPE_THRESHOLD_DEG, 0.0, 1.0)
        return float(score)

    def _score_roughness(self, pts: np.ndarray) -> float:
        """
        Fit a plane to the cell points; score based on RMS of residuals.
        Returns 1.0 for perfectly smooth, 0.0 for roughness >= threshold.
        """
        if len(pts) < 3:
            return 0.0
        centred = pts - pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0
        normal = Vt[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        residuals = np.abs(centred @ normal)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        score = 1.0 - np.clip(rms / ROUGHNESS_THRESHOLD, 0.0, 1.0)
        return float(score)

    def _score_clearance(self, pts: np.ndarray, row: int, col: int) -> float:
        """
        Measure minimum horizontal distance from any high point (obstacle)
        to the cell centre.  "High" = z > 0.3 m (above ground level).
        Returns 1.0 for clear, 0.0 for obstacle within threshold.
        """
        obstacles = pts[pts[:, 2] > 0.3]
        if len(obstacles) == 0:
            return 1.0
        cx, cy = self.cell_to_world_centre(row, col)
        dx = obstacles[:, 0] - cx
        dy = obstacles[:, 1] - cy
        dists = np.sqrt(dx ** 2 + dy ** 2)
        min_dist = float(np.min(dists))
        score = np.clip(
            (min_dist - CLEARANCE_THRESHOLD) / CLEARANCE_THRESHOLD, 0.0, 1.0
        )
        return float(score)
