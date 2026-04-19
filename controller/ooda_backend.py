# controller/ooda_backend.py
# OODABackend — Pegasus Backend implementing the TerraScout OODA loop.
#
# Phase machine: IDLE -> TAKEOFF -> DESCEND_SCAN -> LAND -> ABORT
#
# Critical rules (inherited from dronos):
#   - Do NOT call super().__init__()
#   - self._vehicle = None until initialize(vehicle) is called
#   - All control output goes via self.input_ref (returned by input_reference())
#   - force_and_torques_to_velocities() returns values — does NOT apply them

import numpy as np

from pegasus.simulator.logic.backends.backend import Backend
from controller.decide import Phase, transition_phase
from controller.pid_controller import PIDController
from controller.observe import get_state
from controller.act import compute_rotor_velocities
from controller.terrain_classifier import TerrainClassifier
from terrain.zone_manager import ZoneManager

# ── Tunables ──────────────────────────────────────────────────────────────────
SCAN_ALTITUDE          = 15.0   # metres — TAKEOFF target altitude
TAKEOFF_TOLERANCE      = 0.30   # metres — altitude error to consider hover stable
DESCEND_RATE           = 0.2    # m/s — nominal descent rate during DESCEND_SCAN
DESCEND_TRANSITION_ALT = 8.0    # m — below this + safe zone found -> LAND
DESCEND_ABORT_ALT      = 1.5    # m — below this with no safe zone -> ABORT
LAND_ALT_ZERO          = 0.05   # metres — below this -> touchdown detected

LIDAR_SCAN_EVERY_N     = 3      # call synthetic scan every N update ticks


class OODABackend(Backend):
    """
    Autonomous terrain-scanning OODA loop for TerraScout.

    Phase machine: IDLE -> TAKEOFF -> DESCEND_SCAN -> LAND -> ABORT

    LiDAR data comes from _synthetic_lidar_scan(): analytical raycast against
    the terrain heightmap + boulder registry.  Called every LIDAR_SCAN_EVERY_N
    ticks during DESCEND_SCAN to avoid per-frame overhead.
    """

    def __init__(self, zone_manager: ZoneManager, scan_altitude: float = SCAN_ALTITUDE):
        # Do NOT call super().__init__() — Pegasus rule
        self._vehicle = None
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

        self.zone_manager = zone_manager
        self.scan_altitude = scan_altitude

        # Phase machine
        self._phase = Phase.IDLE
        self._phase_timer = 0.0
        self._tick_count  = 0

        # Sub-systems
        self._pid = PIDController()
        self._classifier = TerrainClassifier(
            terrain_size=zone_manager.terrain_size,
            cell_size=1.0,
        )
        self._mpc = None  # loaded lazily at TAKEOFF exit

        # Selected landing zone (world x, y) — committed in DESCEND_SCAN
        self._target_zone = None

        # State snapshot updated each tick
        self._state = None
        self._received_state = False

        # LiDAR frame counters
        self._lidar_frame_count = 0   # synthetic scan frames fed to classifier

        print("[OODABackend] Initialised.")

    # ── Pegasus Backend interface ─────────────────────────────────────────────

    def initialize(self, vehicle):
        self._vehicle = vehicle
        print("[OODABackend] Vehicle registered.")

    def update_state(self, state):
        self._state = state
        self._received_state = True

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def input_reference(self):
        return self.input_ref

    def start(self):
        print("[OODABackend] start()")

    def stop(self):
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self._phase = Phase.IDLE
        self._phase_timer = 0.0
        self._tick_count  = 0
        self._pid.reset()
        self._classifier.clear()
        self.zone_manager.reset()
        self._target_zone = None
        self._lidar_frame_count = 0
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        print("[OODABackend] reset()")

    # ── Synthetic LiDAR ───────────────────────────────────────────────────────

    def _synthetic_lidar_scan(self, drone_pos: np.ndarray) -> np.ndarray:
        """
        Analytical raycast against terrain heightmap + boulder registry.
        Simulates a downward-looking LiDAR from drone_pos.
        Returns (N, 3) XYZ point cloud in world frame.

        FOV half-angle ~34 deg (fov_radius = alt * 0.6).
        Ray step 0.4 m gives ~(1.2*alt/0.4)^2 points per scan.
        Gaussian noise std=0.02 m matches the RTX LiDAR spec.
        Fully vectorised — no Python loop over ray grid.
        """
        from terrain.terrain_generator import BOULDER_LIST

        alt = float(drone_pos[2])
        fov_radius = alt * 0.6
        step = 0.4

        xs = np.arange(drone_pos[0] - fov_radius,
                       drone_pos[0] + fov_radius + step, step, dtype=np.float32)
        ys = np.arange(drone_pos[1] - fov_radius,
                       drone_pos[1] + fov_radius + step, step, dtype=np.float32)

        if len(xs) == 0 or len(ys) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        gx, gy = np.meshgrid(xs, ys)
        wx = gx.ravel()
        wy = gy.ravel()

        # Vectorised heightmap — same formula as _heightmap_z(), broadcast over array
        wz = (0.40 * np.sin(1.05 * wx + 0.9) * np.cos(0.90 * wy + 0.4)
            + 0.30 * np.cos(1.57 * wx - 0.5) * np.sin(1.40 * wy + 1.2)
            + 0.20 * np.sin(3.14 * wx + 1.7) * np.cos(2.80 * wy - 0.8)
            + 0.40)
        wz = np.clip(wz, -0.5, 1.5)

        # Vectorised boulder spherical-cap heights
        wz_boulder = np.zeros(len(wx), dtype=np.float32)
        for b in BOULDER_LIST:
            dist2 = (wx - b["cx"]) ** 2 + (wy - b["cy"]) ** 2
            mask = dist2 < b["r"] ** 2
            if mask.any():
                cap = np.where(mask, 1.0 - dist2 / (b["r"] ** 2), 0.0)
                wz_boulder = np.maximum(wz_boulder, b["r"] * np.sqrt(np.maximum(cap, 0.0)))

        wz = (wz + wz_boulder).astype(np.float32)
        wz += np.random.normal(0, 0.02, len(wx)).astype(np.float32)

        return np.column_stack([wx, wy, wz]).astype(np.float32)

    # ── Main update tick ──────────────────────────────────────────────────────

    def update(self, dt: float):
        if not self._received_state or self._vehicle is None:
            return

        self._tick_count += 1
        self._phase_timer += dt
        s = get_state(self._vehicle)

        # Synthetic LiDAR scan every LIDAR_SCAN_EVERY_N ticks during DESCEND_SCAN
        if self._phase == Phase.DESCEND_SCAN and self._tick_count % LIDAR_SCAN_EVERY_N == 0:
            pts = self._synthetic_lidar_scan(s["position"])
            if len(pts) > 0:
                self._classifier.add_scan(pts)
                self._lidar_frame_count += 1

        if self._phase == Phase.IDLE:
            self._phase_idle(s)
        elif self._phase == Phase.TAKEOFF:
            self._phase_takeoff(s, dt)
        elif self._phase == Phase.DESCEND_SCAN:
            self._phase_descend_scan(s, dt)
        elif self._phase == Phase.LAND:
            self._phase_land(s, dt)
        elif self._phase == Phase.ABORT:
            self._phase_abort(s, dt)

    # ── Phase implementations ─────────────────────────────────────────────────

    def _phase_idle(self, s):
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self._pid.reset()
        self._classifier.clear()
        self._phase = transition_phase(self._phase, Phase.TAKEOFF)
        self._phase_timer = 0.0

    def _phase_takeoff(self, s, dt):
        target = np.array([0.0, 0.0, self.scan_altitude])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(self._vehicle, float(force), torque)

        alt_err = abs(s["position"][2] - self.scan_altitude)
        speed = np.linalg.norm(s["velocity"])
        if alt_err < TAKEOFF_TOLERANCE and speed < 0.5:
            try:
                from controller.mpc_descent import MPCDescentOptimizer
                self._mpc = MPCDescentOptimizer()
                print("[OODABackend] MPC loaded for descent.")
            except ImportError:
                self._mpc = None
                print("[OODABackend] MPC not available — using PID descent.")
            self._phase = transition_phase(self._phase, Phase.DESCEND_SCAN)
            self._phase_timer = 0.0

    def _phase_descend_scan(self, s, dt):
        """
        Descend at DESCEND_RATE m/s while classifying terrain below.
        MPC replans lateral position toward the current best zone each tick.
        """
        alt = float(s["position"][2])

        # Periodic status print every 2 seconds
        if int(self._phase_timer / 2.0) > int((self._phase_timer - dt) / 2.0):
            best = self.zone_manager.best_zone()
            if best is not None:
                best_str = f"P={best[2]:.3f} at ({best[0]:.1f},{best[1]:.1f})"
            else:
                best_str = "none"
            print(f"[DESCEND_SCAN] alt={alt:.1f}m  "
                  f"scans={self._lidar_frame_count}  "
                  f"coverage={self.zone_manager.coverage_fraction():.1%}  "
                  f"best={best_str}")

        # Abort if too low with still no safe zone
        if alt < DESCEND_ABORT_ALT and self.zone_manager.best_zone() is None:
            cov = self.zone_manager.coverage_fraction()
            n_safe = self.zone_manager.safe_zone_count()
            print(f"\n{'!' * 50}")
            print(f"[ABORT] No safe zone found after full descent")
            print(f"  Coverage: {cov:.1%}  Safe zones: {n_safe}")
            print(f"  Returning to origin")
            print(f"{'!' * 50}\n")
            self._phase = transition_phase(self._phase, Phase.ABORT)
            self._phase_timer = 0.0
            return

        # Push classifier results into zone_manager every tick
        results = self._classifier.get_all_results()
        for (row, col), res in results.items():
            wx, wy = self._classifier.cell_to_world_centre(row, col)
            self.zone_manager.update_cell(wx, wy, res["p_safe"], res["variance"])

        # Transition to LAND when a safe zone exists and we're low enough
        best = self.zone_manager.best_zone()
        if best is not None and alt < DESCEND_TRANSITION_ALT:
            bx, by, bscore = best
            # Look up Kalman variance for the selected cell
            cell = self.zone_manager.world_to_cell(bx, by)
            var = float(self.zone_manager._variances[cell[0], cell[1]]) if cell is not None else float("nan")
            pos = s["position"]
            print(f"\n{'=' * 50}")
            print(f"[OODA] ZONE SELECTED: ({bx:.1f}, {by:.1f})")
            print(f"       P(safe)={bscore:.3f}  Kalman_var={var:.4f}")
            print(f"       Drone at ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
            print(f"{'=' * 50}\n")
            self._target_zone = (bx, by)
            self._phase = transition_phase(self._phase, Phase.LAND)
            self._phase_timer = 0.0
            return

        # Lateral target: track best zone estimate, or hold origin
        tx, ty = (best[0], best[1]) if best is not None else (0.0, 0.0)
        target_z = max(DESCEND_ABORT_ALT, alt - DESCEND_RATE * dt)
        target = np.array([tx, ty, target_z])

        if self._mpc is not None:
            target_state = np.array([tx, ty, target_z, 0.0, 0.0, -DESCEND_RATE])
            current_state = np.concatenate([s["position"], s["velocity"]])
            u_opt = self._mpc.compute(current_state, target_state)
            force = float(1.50 * (float(u_opt[2]) + 9.81))
            _, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)
        else:
            force, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, float(force), torque)

    def _phase_land(self, s, dt):
        """Final descent to committed zone. MPC if available, else PID."""
        if self._target_zone is None:
            self._phase = transition_phase(self._phase, Phase.ABORT)
            return

        zx, zy = self._target_zone

        alt = float(s["position"][2])
        lat_err = float(np.sqrt((s["position"][0] - zx) ** 2 + (s["position"][1] - zy) ** 2))
        vz = float(s["velocity"][2])
        using_mpc = self._mpc is not None
        print(f"[LAND] alt={alt:.2f}m  lateral_err={lat_err:.2f}m  vz={vz:.2f}m/s  {'MPC' if using_mpc else 'PID'}")

        if alt < LAND_ALT_ZERO:
            fx, fy, fz = float(s["position"][0]), float(s["position"][1]), float(s["position"][2])
            final_lat = float(np.sqrt((fx - zx) ** 2 + (fy - zy) ** 2))
            print(f"\n{'*' * 50}")
            print(f"[TOUCHDOWN] Mission complete!")
            print(f"  Landing zone: ({zx:.1f}, {zy:.1f})")
            print(f"  Final position: ({fx:.1f}, {fy:.1f}, {fz:.2f})")
            print(f"  Lateral error: {final_lat:.3f}m")
            print(f"  Total LiDAR scans: {self._lidar_frame_count}")
            print(f"{'*' * 50}\n")
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            return

        target = np.array([zx, zy, 0.0])
        if self._mpc is not None:
            target_state = np.array([zx, zy, 0.0, 0.0, 0.0, 0.0])
            current_state = np.concatenate([s["position"], s["velocity"]])
            u_opt = self._mpc.compute(current_state, target_state)
            force = float(1.50 * (float(u_opt[2]) + 9.81))
            _, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)
        else:
            force, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, float(force), torque)

    def _phase_abort(self, s, dt):
        """No safe zone — descend to origin."""
        if s["position"][2] < LAND_ALT_ZERO:
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            return
        target = np.array([0.0, 0.0, max(0.0, s["position"][2] - 0.5)])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(self._vehicle, float(force), torque)

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def target_zone(self):
        return self._target_zone
