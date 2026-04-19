# scanner/scan_planner.py
# Lawnmower scan path generator for TerraScout SCAN phase.
# Produces a list of (x, y, z) waypoints the drone visits during SCAN.
# Pure Python/numpy — no Isaac Sim or ROS2 imports.

import numpy as np
from typing import List, Tuple


def generate_lawnmower(
    terrain_size: float = 10.0,
    altitude: float = 8.0,
    strip_spacing: float = 2.0,
    start_x: float = None,
) -> List[Tuple[float, float, float]]:
    """
    Generate a boustrophedon (lawnmower) scan path over the terrain grid.

    The path runs parallel strips along the Y axis, spaced strip_spacing apart
    in X.  Alternate strips reverse direction so the drone never backtracks.

    Parameters
    ----------
    terrain_size : float
        Half-extent of terrain — scan covers [-terrain_size, terrain_size].
    altitude : float
        Constant scan altitude in metres.
    strip_spacing : float
        Distance between adjacent scan strips in metres.
        Should be ≤ LiDAR half-swath at scan_altitude for full coverage.
    start_x : float, optional
        Starting X position.  Defaults to -terrain_size.

    Returns
    -------
    List of (x, y, z) waypoints.  First waypoint is at (start_x, -terrain_size, alt).
    """
    if start_x is None:
        start_x = -terrain_size

    waypoints = []
    x = start_x
    forward = True   # True = Y increasing, False = Y decreasing

    while x <= terrain_size + 1e-6:
        if forward:
            waypoints.append((x, -terrain_size, altitude))
            waypoints.append((x,  terrain_size, altitude))
        else:
            waypoints.append((x,  terrain_size, altitude))
            waypoints.append((x, -terrain_size, altitude))
        x += strip_spacing
        forward = not forward

    return waypoints


def coverage_at_altitude(altitude: float, lidar_fov_deg: float = 30.0) -> float:
    """
    Estimate the LiDAR ground-swath half-width at a given altitude.
    Used to choose an appropriate strip_spacing.

    LiDAR field of view (default 30° half-angle) gives:
        swath_half = altitude * tan(fov_deg)

    Returns the full swath width in metres.
    """
    return 2.0 * altitude * np.tan(np.radians(lidar_fov_deg))


class ScanPlanner:
    """
    Stateful scan path manager used by OODABackend.

    Usage
    -----
    planner = ScanPlanner(terrain_size=10.0, altitude=8.0)
    planner.reset()
    while not planner.complete:
        target = planner.current_waypoint()
        if distance_to(target) < 0.5:
            planner.advance()
    """

    def __init__(
        self,
        terrain_size: float = 10.0,
        altitude: float = 8.0,
        strip_spacing: float = 2.0,
    ):
        self.terrain_size = terrain_size
        self.altitude = altitude
        self.strip_spacing = strip_spacing
        self._waypoints: List[Tuple[float, float, float]] = []
        self._index = 0

    def reset(self):
        """(Re)generate waypoints and reset progress."""
        self._waypoints = generate_lawnmower(
            terrain_size=self.terrain_size,
            altitude=self.altitude,
            strip_spacing=self.strip_spacing,
        )
        self._index = 0
        print(f"[ScanPlanner] {len(self._waypoints)} waypoints generated, "
              f"strip spacing={self.strip_spacing}m.")

    @property
    def complete(self) -> bool:
        """True once all waypoints have been visited."""
        return self._index >= len(self._waypoints)

    def current_waypoint(self) -> Tuple[float, float, float]:
        """Return the current target waypoint.  Raises if complete."""
        if self.complete:
            raise StopIteration("Scan path complete.")
        return self._waypoints[self._index]

    def advance(self):
        """Mark current waypoint reached; step to next."""
        self._index = min(self._index + 1, len(self._waypoints))
        if not self.complete:
            wp = self._waypoints[self._index]
            print(f"[ScanPlanner] → waypoint {self._index}/{len(self._waypoints)-1} "
                  f"({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")

    def progress(self) -> float:
        """Fraction of waypoints completed, 0.0–1.0."""
        if not self._waypoints:
            return 0.0
        return self._index / len(self._waypoints)
