# controller/lidar_viz.py
# LidarViz — top-down zone-map display in a floating omni.ui window.
#
# Shows a live colour grid of the ZoneManager status updated each call to refresh().
# Call refresh(zone_manager, ooda_backend) from the main sim loop — cheap numpy op.
#
# Colour key:
#   Grey  = UNKNOWN
#   Green = SAFE
#   Yellow = MARGINAL
#   Red   = UNSAFE
#   Cyan dot = latest LiDAR scan footprint (last_scan_pts)
#   White X  = committed landing zone
#
# Uses omni.ui ByteImageProvider — no matplotlib, no external deps, pure numpy.
# Window is 400x400 px. Each cell is (400 / grid_n) px wide.

import numpy as np

try:
    import omni.ui as ui
    _OMNI_AVAILABLE = True
except ImportError:
    _OMNI_AVAILABLE = False


class LidarViz:
    """
    Floating omni.ui window showing the live zone grid + scan footprint.

    Parameters
    ----------
    terrain_size : float
        Half-extent of terrain (metres). Grid covers ±terrain_size.
    cell_size : float
        Cell size in metres (must match ZoneManager).
    px : int
        Window width/height in pixels (square).
    update_every : int
        Refresh the display every N calls to refresh(). Reduces GPU upload rate.
    """

    # Status colours (RGBA uint8)
    _COLORS = {
        "UNKNOWN":  (100, 100, 100, 255),
        "SAFE":     (30,  200, 60,  255),
        "MARGINAL": (220, 200, 30,  255),
        "UNSAFE":   (200, 40,  40,  255),
    }
    _SCAN_COLOR   = (0,   220, 220, 180)   # cyan — recent scan footprint
    _TARGET_COLOR = (255, 255, 255, 255)   # white X — committed landing zone
    _DRONE_COLOR  = (255, 165,   0, 255)   # orange dot — drone current position

    def __init__(self, terrain_size: float, cell_size: float = 0.5,
                 px: int = 400, update_every: int = 10):
        self.terrain_size = terrain_size
        self.cell_size    = cell_size
        self.px           = px
        self.update_every = update_every
        self._call_count  = 0

        n = int(2 * terrain_size / cell_size)
        self._n   = n
        self._buf = np.zeros((px, px, 4), dtype=np.uint8)

        # Pixels per cell (float — used for mapping)
        self._ppc = px / n

        if not _OMNI_AVAILABLE:
            print("[LidarViz] omni.ui not available — running headless (no window).")
            self._window   = None
            self._provider = None
            return

        self._provider = ui.ByteImageProvider()
        self._provider.set_bytes_data(
            self._buf.flatten().tolist(), [px, px]
        )

        self._window = ui.Window(
            "LiDAR Zone Map", width=px + 20, height=px + 40,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
        )
        with self._window.frame:
            with ui.VStack():
                ui.Label("Zone Map  (G=Safe  Y=Marginal  R=Unsafe  C=Scan)",
                         height=18, style={"color": 0xFFCCCCCC, "font_size": 11})
                ui.ImageWithProvider(
                    self._provider, width=px, height=px,
                )
        print(f"[LidarViz] Window opened ({px}x{px}px, {n}x{n} cells @ {cell_size}m).")

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self, zone_manager, ooda_backend=None):
        """
        Redraw the zone grid.  Call from the main sim loop (cheap — skips most frames).

        Parameters
        ----------
        zone_manager  : ZoneManager
        ooda_backend  : OODABackend (optional) — for scan footprint + target overlay
        """
        self._call_count += 1
        if self._call_count % self.update_every != 0:
            return
        if self._provider is None and _OMNI_AVAILABLE:
            return

        self._draw_grid(zone_manager)

        if ooda_backend is not None:
            self._draw_scan_footprint(ooda_backend)
            self._draw_drone(ooda_backend)
            self._draw_target(ooda_backend)

        if _OMNI_AVAILABLE and self._provider is not None:
            self._provider.set_bytes_data(
                self._buf.flatten().tolist(),
                [self.px, self.px],
            )

    def close(self):
        if _OMNI_AVAILABLE and self._window is not None:
            self._window.destroy()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _world_to_px(self, wx: float, wy: float):
        """World (x, y) → pixel (col, row).  Row 0 = top = max Y."""
        col = int((wx + self.terrain_size) / self.cell_size * self._ppc)
        row = int((self.terrain_size - wy) / self.cell_size * self._ppc)
        return col, row

    def _draw_grid(self, zone_manager):
        """Fill background with status colours."""
        buf = self._buf
        n   = self._n
        ppc = self._ppc

        for row_c in range(n):
            for col_c in range(n):
                status = zone_manager._status[row_c, col_c]
                name = status.name if hasattr(status, 'name') else str(status)
                color = self._COLORS.get(name, self._COLORS["UNKNOWN"])
                # Pixel rectangle for this cell
                px0 = int(col_c * ppc)
                px1 = int((col_c + 1) * ppc)
                # Flip Y: row 0 of grid = bottom of terrain = bottom of image → top row = high Y
                py0 = int((n - 1 - row_c) * ppc)
                py1 = int((n - row_c) * ppc)
                buf[py0:py1, px0:px1] = color

        # Grid lines — thin dark separator every cell
        step_px = max(1, int(ppc * 5))
        buf[::step_px, :, :3] = 20
        buf[:, ::step_px, :3] = 20

    def _draw_scan_footprint(self, ooda_backend):
        """Draw latest scan points as small cyan squares."""
        pts = getattr(ooda_backend, '_last_scan_pts', None)
        if pts is None or len(pts) == 0:
            return
        # Subsample to avoid drawing every point (expensive for large scans)
        idx = np.arange(0, len(pts), max(1, len(pts)//800))
        for wx, wy, _ in pts[idx]:
            col, row = self._world_to_px(float(wx), float(wy))
            r0, r1 = max(0, row-1), min(self.px, row+2)
            c0, c1 = max(0, col-1), min(self.px, col+2)
            self._buf[r0:r1, c0:c1] = self._SCAN_COLOR

    def _draw_drone(self, ooda_backend):
        """Draw drone world position as a 5x5 orange square."""
        state = getattr(ooda_backend, '_state', None)
        if state is None:
            return
        # _state may be a Pegasus State object (.position) or a dict ("position")
        if isinstance(state, dict):
            pos = state.get('position')
        else:
            pos = getattr(state, 'position', None)
        if pos is None or len(pos) < 2:
            return
        col, row = self._world_to_px(float(pos[0]), float(pos[1]))
        arm = 2  # 5x5 square: center ±2 pixels
        r0 = max(0, row - arm); r1 = min(self.px, row + arm + 1)
        c0 = max(0, col - arm); c1 = min(self.px, col + arm + 1)
        self._buf[r0:r1, c0:c1] = self._DRONE_COLOR

    def _draw_target(self, ooda_backend):
        """Draw committed landing zone as a white cross."""
        target = getattr(ooda_backend, '_target_zone', None)
        if target is None:
            return
        wx, wy = target
        col, row = self._world_to_px(float(wx), float(wy))
        arm = max(3, int(self._ppc * 1.5))
        # Horizontal bar
        r0 = max(0, row - 1); r1 = min(self.px, row + 2)
        c0 = max(0, col - arm); c1 = min(self.px, col + arm)
        self._buf[r0:r1, c0:c1] = self._TARGET_COLOR
        # Vertical bar
        r0 = max(0, row - arm); r1 = min(self.px, row + arm)
        c0 = max(0, col - 1);   c1 = min(self.px, col + 2)
        self._buf[r0:r1, c0:c1] = self._TARGET_COLOR
