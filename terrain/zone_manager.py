# terrain/zone_manager.py
# ZoneManager — terrain cell registry for TerraScout.
# Replaces dronos BoxManager.  No USD interaction — pure data structure.
#
# Coordinate convention:
#   World origin (0,0) maps to grid centre.
#   Cell (r, c) covers world X in [x0 + c*cell_size, x0 + (c+1)*cell_size]
#   and world Y in [y0 + r*cell_size, y0 + (r+1)*cell_size].

import numpy as np
from enum import Enum
from scipy.ndimage import label


class ZoneStatus(Enum):
    UNKNOWN  = 0   # not yet observed by LiDAR
    SAFE     = 1   # score > SAFE_THRESHOLD — viable landing zone
    MARGINAL = 2   # score in [MARGINAL_THRESHOLD, SAFE_THRESHOLD]
    UNSAFE   = 3   # score < MARGINAL_THRESHOLD — rejected


SAFE_THRESHOLD     = 0.55
MARGINAL_THRESHOLD = 0.35


class ZoneManager:
    """
    Maintains a 2-D grid of terrain cells and their safety scores.

    Parameters
    ----------
    terrain_size : float
        Half-extent of the terrain in metres (grid spans ±terrain_size).
    cell_size : float
        Width/height of each grid cell in metres.
    """

    def __init__(self, terrain_size: float = 10.0, cell_size: float = 1.0):
        self.terrain_size = terrain_size
        self.cell_size = cell_size

        n = int(2 * terrain_size / cell_size)
        self.grid_w = n
        self.grid_h = n

        # Score grid — NaN = not yet observed
        self._scores = np.full((n, n), np.nan, dtype=np.float32)
        # Kalman variance grid — NaN = not yet observed
        self._variances = np.full((n, n), np.nan, dtype=np.float32)
        # Status grid
        self._status = np.full((n, n), ZoneStatus.UNKNOWN, dtype=object)

        # World coordinate of grid origin (bottom-left corner)
        self._origin_x = -terrain_size
        self._origin_y = -terrain_size

        print(f"[ZoneManager] Grid {n}x{n}, cell_size={cell_size}m, "
              f"covers [{-terrain_size}, {terrain_size}] m in X and Y.")

    # ── Cell addressing ───────────────────────────────────────────────────────

    def world_to_cell(self, wx: float, wy: float):
        """Convert world (x, y) to grid (row, col). Returns None if out of bounds."""
        col = int((wx - self._origin_x) / self.cell_size)
        row = int((wy - self._origin_y) / self.cell_size)
        if 0 <= row < self.grid_h and 0 <= col < self.grid_w:
            return row, col
        return None

    def cell_to_world_centre(self, row: int, col: int):
        """Return world (x, y) at the centre of cell (row, col)."""
        cx = self._origin_x + (col + 0.5) * self.cell_size
        cy = self._origin_y + (row + 0.5) * self.cell_size
        return cx, cy

    # ── Update ────────────────────────────────────────────────────────────────

    def update_cell(self, wx: float, wy: float, p_safe: float, variance: float = 0.0):
        """
        Record a Bayesian P(safe) and Kalman variance for the cell at world position (wx, wy).
        p_safe must be in [0, 1].  Existing scores are averaged to reduce noise.
        """
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return
        row, col = cell
        if np.isnan(self._scores[row, col]):
            self._scores[row, col] = p_safe
        else:
            # Exponential moving average — weight new observations at 0.3
            self._scores[row, col] = 0.7 * self._scores[row, col] + 0.3 * p_safe
        self._variances[row, col] = variance

        # Update status bucket — SAFE requires both P(safe) threshold AND low variance
        s = float(self._scores[row, col])
        if s >= SAFE_THRESHOLD and variance < 0.05:
            self._status[row, col] = ZoneStatus.SAFE
        elif s >= MARGINAL_THRESHOLD:
            self._status[row, col] = ZoneStatus.MARGINAL
        else:
            self._status[row, col] = ZoneStatus.UNSAFE

    # ── Queries ───────────────────────────────────────────────────────────────

    def best_zone(self, margin: float = 2.0):
        """
        Return (wx, wy, score) of the highest-scoring SAFE cell whose world
        coordinates are at least `margin` metres inside the terrain boundary.
        Returns None if no qualifying cell has reached SAFE status yet.
        """
        limit = self.terrain_size - margin
        best_score = -1.0
        best_cell = None
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                if self._status[row, col] == ZoneStatus.SAFE:
                    wx, wy = self.cell_to_world_centre(row, col)
                    if abs(wx) > limit or abs(wy) > limit:
                        continue   # too close to terrain edge
                    s = float(self._scores[row, col])
                    if s > best_score:
                        best_score = s
                        best_cell = (row, col)
        if best_cell is None:
            return None
        row, col = best_cell
        wx, wy = self.cell_to_world_centre(row, col)
        return wx, wy, best_score

    def coverage_fraction(self) -> float:
        """Fraction of cells that have been observed (not UNKNOWN)."""
        observed = np.sum(~np.isnan(self._scores))
        total = self.grid_w * self.grid_h
        return float(observed) / float(total)

    def safe_zone_count(self) -> int:
        """Number of cells currently classified as SAFE."""
        return int(np.sum(self._status == ZoneStatus.SAFE))

    def score_at(self, wx: float, wy: float):
        """Return score for cell at world (wx, wy), or None if not observed."""
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return None
        row, col = cell
        v = self._scores[row, col]
        return None if np.isnan(v) else float(v)

    def status_at(self, wx: float, wy: float) -> ZoneStatus:
        """Return ZoneStatus for cell at world (wx, wy)."""
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return ZoneStatus.UNKNOWN
        row, col = cell
        return self._status[row, col]

    def all_safe_zones(self):
        """Yield (wx, wy, score) for every SAFE cell."""
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                if self._status[row, col] == ZoneStatus.SAFE:
                    wx, wy = self.cell_to_world_centre(row, col)
                    yield wx, wy, float(self._scores[row, col])

    def _find_clusters(self, min_cells: int, margin: float):
        """
        Internal: find all contiguous SAFE clusters that meet the minimum cell count
        and are within the terrain margin.

        Uses scipy.ndimage.label for connected-component analysis.
        Returns list of (cx_world, cy_world, mean_score, size) sorted by score desc.
        """
        limit = self.terrain_size - margin

        # Build binary safe grid with margin filter applied
        safe_grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if self._status[r, c] == ZoneStatus.SAFE:
                    wx, wy = self.cell_to_world_centre(r, c)
                    if abs(wx) <= limit and abs(wy) <= limit:
                        safe_grid[r, c] = True

        labeled, n_features = label(safe_grid)
        if n_features == 0:
            return []

        clusters = []
        for cluster_id in range(1, n_features + 1):
            cluster_mask = labeled == cluster_id
            size = int(cluster_mask.sum())
            if size < min_cells:
                continue
            rows, cols = np.where(cluster_mask)
            scores = [float(self._scores[r, c])
                      for r, c in zip(rows, cols)
                      if not np.isnan(self._scores[r, c])]
            if not scores:
                continue
            mean_score = float(np.mean(scores))
            cx = float(np.mean([self.cell_to_world_centre(r, c)[0]
                                 for r, c in zip(rows, cols)]))
            cy = float(np.mean([self.cell_to_world_centre(r, c)[1]
                                 for r, c in zip(rows, cols)]))
            clusters.append((cx, cy, mean_score, size))

        return sorted(clusters, key=lambda x: -x[2])

    def best_landing_zone(self, min_radius_m: float = 1.0, margin: float = 2.0):
        """
        Find best landing zone as a contiguous cluster of SAFE cells.

        Requires a minimum contiguous safe area (~pi * min_radius_m^2 cells).
        Single isolated SAFE cells are not sufficient — the drone needs a real
        flat patch, not a noise spike.

        Parameters
        ----------
        min_radius_m : float
            Minimum radius of contiguous safe area in metres.
            At cell_size=1m, 1.5m radius ≈ 7 cells (pi*1.5^2).
        margin : float
            Reject clusters whose centre is within this distance of the boundary.

        Returns
        -------
        (cx, cy, score) of the best cluster centre, or None.
        """
        min_cells = max(1, int(np.pi * (min_radius_m / self.cell_size) ** 2))
        clusters = self._find_clusters(min_cells, margin)
        if not clusters:
            return None
        cx, cy, score, size = clusters[0]
        return cx, cy, score

    def get_all_clusters(self, min_radius_m: float = 1.0, margin: float = 2.0):
        """
        Return list of (cx, cy, score, size) for every valid landing cluster,
        sorted by mean score descending.  Used for diagnostic printing.
        """
        min_cells = max(1, int(np.pi * (min_radius_m / self.cell_size) ** 2))
        return self._find_clusters(min_cells, margin)

    def reset(self):
        """Clear all observations — used when restarting a scan."""
        self._scores[:] = np.nan
        self._variances[:] = np.nan
        self._status[:] = ZoneStatus.UNKNOWN
        print("[ZoneManager] Reset — all cells UNKNOWN.")
