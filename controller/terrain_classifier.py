# controller/terrain_classifier.py
# TerrainClassifier — Bayesian + Kalman terrain safety classifier.
#
# Pipeline per cell:
#   Stage 1: Geometric pre-filter  — slope > 30 deg → UNSAFE immediately
#   Stage 2: Kalman height tracker — estimates mean height, gates on variance
#   Stage 3: Bayesian posterior    — P(safe | slope, roughness) via log-likelihood
#
# Covers course modules M3 (Kalman variance gate), M6 (Gaussian MLE likelihood),
# M7 (Bayesian update), M9 (Kalman filter).
#
# Calls scanner.lidar_classifier.LidarClassifier internally.
# Do NOT modify lidar_classifier.py.

import numpy as np
from scipy.stats import norm

from scanner.lidar_classifier import LidarClassifier


# ── Kalman parameters ─────────────────────────────────────────────────────────
KALMAN_Q = 0.001        # process noise variance (static terrain — small)
KALMAN_R = 0.02 ** 2   # measurement noise variance (LiDAR z-noise ≈ 2 cm)
KALMAN_VAR_INIT = 1.0   # initial (high) estimate variance — converges over time
KALMAN_VAR_GATE = 0.05  # m² — only run Bayesian once variance below this

# ── Bayesian likelihood parameters ───────────────────────────────────────────
# P(slope | safe)   ~ N(mean=0 deg, std=5 deg)
# P(slope | unsafe) ~ N(mean=20 deg, std=8 deg)
SLOPE_SAFE_MEAN,   SLOPE_SAFE_STD   =  0.0, 5.0
SLOPE_UNSAFE_MEAN, SLOPE_UNSAFE_STD = 20.0, 8.0

# P(roughness | safe)   ~ N(mean=0 m, std=0.05 m)
# P(roughness | unsafe) ~ N(mean=0.2 m, std=0.08 m)
ROUGH_SAFE_MEAN,   ROUGH_SAFE_STD   = 0.00, 0.05
ROUGH_UNSAFE_MEAN, ROUGH_UNSAFE_STD = 0.20, 0.08

# ── Classification thresholds ─────────────────────────────────────────────────
SLOPE_PREFILTER_DEG = 30.0   # immediate UNSAFE — skip Bayesian
P_SAFE_THRESHOLD    = 0.85   # P(safe) above this → SAFE (also needs var < gate)
P_UNSAFE_THRESHOLD  = 0.20   # P(safe) below this → UNSAFE
PRIOR_P_SAFE        = 0.50   # uninformative prior


class _CellState:
    """Per-cell Kalman state + Bayesian posterior."""

    __slots__ = ("z_bar", "P", "p_safe", "n_obs", "status")

    def __init__(self):
        self.z_bar  = 0.0             # Kalman mean height estimate [m]
        self.P      = KALMAN_VAR_INIT # Kalman variance [m²]
        self.p_safe = PRIOR_P_SAFE    # Bayesian P(safe)
        self.n_obs  = 0               # number of LiDAR frames that hit this cell
        self.status = "UNCERTAIN"     # "SAFE" | "UNSAFE" | "UNCERTAIN"


class TerrainClassifier:
    """
    Full terrain safety classifier: geometric pre-filter → Kalman → Bayesian.

    Parameters
    ----------
    terrain_size : float
        Half-extent of terrain in metres (grid covers ±terrain_size).
    cell_size : float
        Width/height of each grid cell in metres.
    """

    def __init__(self, terrain_size: float, cell_size: float = 1.0):
        self.terrain_size = terrain_size
        self.cell_size = cell_size

        # Delegate point accumulation + geometric feature extraction to lidar_classifier
        self._lidar = LidarClassifier(
            cell_size=cell_size,
            terrain_size=terrain_size,
        )

        n = int(2 * terrain_size / cell_size)
        self._n = n
        self._origin = -terrain_size

        # Per-cell state: (row, col) -> _CellState
        self._cells: dict = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def add_scan(self, points: np.ndarray):
        """
        Process one LiDAR frame.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3) or (N, 4)
            XYZ (+ optional intensity) in world frame.
        """
        # Accumulate raw points into the underlying lidar_classifier
        self._lidar.add_points(points)

        # Derive per-cell slope/roughness and run the full pipeline
        scores = self._lidar.compute_scores()   # {(row,col): score_float}

        # We also need the raw geometric features, not just the composite score.
        # Re-derive slope and roughness from lidar internals where available.
        cell_points = self._lidar._cell_points

        for (row, col), pts_list in cell_points.items():
            pts = np.array(pts_list, dtype=np.float32)
            if len(pts) < 5:
                continue

            slope_deg, roughness_rms = self._extract_features(pts)
            self._update_cell(row, col, pts, slope_deg, roughness_rms)

    def get_cell_result(self, row: int, col: int) -> dict:
        """
        Return classification result for one cell.

        Returns
        -------
        dict with keys: p_safe, variance, status, n_obs
        """
        if (row, col) not in self._cells:
            return {
                "p_safe":   PRIOR_P_SAFE,
                "variance": KALMAN_VAR_INIT,
                "status":   "UNCERTAIN",
                "n_obs":    0,
            }
        cs = self._cells[(row, col)]
        return {
            "p_safe":   cs.p_safe,
            "variance": cs.P,
            "status":   cs.status,
            "n_obs":    cs.n_obs,
        }

    def get_all_results(self) -> dict:
        """
        Return classification results for all observed cells.

        Returns
        -------
        dict mapping (row, col) -> result_dict (same format as get_cell_result)
        """
        return {key: self.get_cell_result(*key) for key in self._cells}

    def cell_to_world_centre(self, row: int, col: int):
        """Return world (x, y) at the centre of cell (row, col)."""
        cx = self._origin + (col + 0.5) * self.cell_size
        cy = self._origin + (row + 0.5) * self.cell_size
        return cx, cy

    def kalman_converged(self, row: int, col: int) -> bool:
        """True when Kalman variance has fallen below the gate threshold."""
        if (row, col) not in self._cells:
            return False
        return self._cells[(row, col)].P < KALMAN_VAR_GATE

    def clear(self):
        """Reset for a new scan pass — clears all cells and the underlying lidar buffer."""
        self._cells.clear()
        self._lidar.clear()

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _update_cell(
        self,
        row: int,
        col: int,
        pts: np.ndarray,
        slope_deg: float,
        roughness_rms: float,
    ):
        """Run stages 1–3 for one cell and update its _CellState."""
        if (row, col) not in self._cells:
            self._cells[(row, col)] = _CellState()

        cs = self._cells[(row, col)]
        cs.n_obs += 1

        # ── Stage 1: Geometric pre-filter ─────────────────────────────────────
        if slope_deg > SLOPE_PREFILTER_DEG:
            cs.p_safe  = 0.0
            cs.status  = "UNSAFE"
            return   # skip Kalman + Bayesian for steep cells

        # ── Stage 2: Kalman height tracker ───────────────────────────────────
        z_meas = float(np.mean(pts[:, 2]))   # measured mean height this frame

        # Predict
        cs.P = cs.P + KALMAN_Q

        # Update
        K      = cs.P / (cs.P + KALMAN_R)
        cs.z_bar = cs.z_bar + K * (z_meas - cs.z_bar)
        cs.P   = (1.0 - K) * cs.P

        # Gate: only classify if variance has converged
        if cs.P >= KALMAN_VAR_GATE:
            cs.status = "UNCERTAIN"
            return

        # ── Stage 3: Bayesian posterior ───────────────────────────────────────
        # Log-likelihoods for numerical stability
        log_lk_safe = (
            norm.logpdf(slope_deg,    SLOPE_SAFE_MEAN,   SLOPE_SAFE_STD) +
            norm.logpdf(roughness_rms, ROUGH_SAFE_MEAN,  ROUGH_SAFE_STD)
        )
        log_lk_unsafe = (
            norm.logpdf(slope_deg,    SLOPE_UNSAFE_MEAN, SLOPE_UNSAFE_STD) +
            norm.logpdf(roughness_rms, ROUGH_UNSAFE_MEAN, ROUGH_UNSAFE_STD)
        )

        log_prior_safe   = np.log(cs.p_safe   + 1e-12)
        log_prior_unsafe = np.log(1.0 - cs.p_safe + 1e-12)

        log_post_safe   = log_prior_safe   + log_lk_safe
        log_post_unsafe = log_prior_unsafe + log_lk_unsafe

        # Normalise in log-space (log-sum-exp)
        log_max = max(log_post_safe, log_post_unsafe)
        log_norm = log_max + np.log(
            np.exp(log_post_safe - log_max) + np.exp(log_post_unsafe - log_max)
        )
        cs.p_safe = float(np.exp(log_post_safe - log_norm))
        cs.p_safe = float(np.clip(cs.p_safe, 0.0, 1.0))

        # ── Classify ──────────────────────────────────────────────────────────
        if cs.p_safe >= P_SAFE_THRESHOLD and cs.P < KALMAN_VAR_GATE:
            cs.status = "SAFE"
        elif cs.p_safe <= P_UNSAFE_THRESHOLD:
            cs.status = "UNSAFE"
        else:
            cs.status = "UNCERTAIN"

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(pts: np.ndarray):
        """
        Derive slope (degrees) and roughness (RMS plane residual) from a cell's
        point cloud via PCA — mirrors lidar_classifier internals exactly.

        Returns (slope_deg, roughness_rms).
        """
        if len(pts) < 3:
            return 90.0, 1.0   # treat as worst-case if too few points

        centred = pts - pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            return 90.0, 1.0

        normal = Vt[-1]
        if normal[2] < 0:
            normal = -normal
        normal_unit = normal / (np.linalg.norm(normal) + 1e-9)

        # Slope: angle between surface normal and vertical
        cos_angle = float(np.clip(normal_unit[2], 0.0, 1.0))
        slope_deg = float(np.degrees(np.arccos(cos_angle)))

        # Roughness: RMS of plane residuals
        residuals  = np.abs(centred @ normal_unit)
        roughness  = float(np.sqrt(np.mean(residuals ** 2)))

        return slope_deg, roughness
