# controller/ooda_backend.py
# OODABackend — Pegasus Backend subclass implementing the TerraScout OODA loop.
#
# Phase machine: IDLE → TAKEOFF → SCAN → ASSESS → APPROACH → LAND
#
# Critical rules (inherited from dronos):
#   - Do NOT call super().__init__()
#   - self._vehicle = None until initialize(vehicle) is called
#   - All control output goes via self.input_ref (returned by input_reference())
#   - force_and_torques_to_velocities() returns values — does NOT apply them

import numpy as np
import time

from pegasus.simulator.logic.backends.backend import Backend
from controller.decide import Phase, transition_phase
from controller.pid_controller import PIDController
from controller.observe import get_state
from controller.act import compute_rotor_velocities
from controller.terrain_classifier import TerrainClassifier
from scanner.scan_planner import ScanPlanner
from terrain.zone_manager import ZoneManager

# ── Tunables ──────────────────────────────────────────────────────────────────
SCAN_ALTITUDE         = 30.0   # metres — TAKEOFF target + SCAN cruise height
TAKEOFF_TOLERANCE     = 0.30   # metres — altitude error to consider hover stable
WAYPOINT_TOLERANCE    = 0.50   # metres — distance to consider waypoint reached
APPROACH_LAT_TOL      = 0.50   # metres — lateral error to consider above zone
LAND_ALT_ZERO         = 0.05   # metres — below this → touchdown detected
APPROACH_ALTITUDE     = 5.0    # metres — hover altitude above zone before descent
SCAN_STRIP_SPACING    = 2.0    # metres — lawnmower strip spacing
SCAN_COVERAGE_TARGET  = 0.95   # fraction of cells that must be scanned
SAFE_THRESHOLD        = 0.65   # zone score threshold to accept zone
MAX_SCAN_RETRIES      = 2      # expand bbox and retry if no zone found


class OODABackend(Backend):
    """
    Autonomous terrain-scanning OODA loop for TerraScout.

    Parameters
    ----------
    zone_manager : ZoneManager
        Shared zone registry — receives scores from LiDAR classifier.
    scan_altitude : float
        Target altitude for lawnmower scan pass.
    """

    def __init__(self, zone_manager: ZoneManager, scan_altitude: float = SCAN_ALTITUDE):
        # Do NOT call super().__init__() — Pegasus rule
        self._vehicle = None
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

        self.zone_manager = zone_manager
        self.scan_altitude = scan_altitude

        # Phase machine
        self._phase = Phase.IDLE
        self._phase_timer = 0.0         # seconds spent in current phase

        # Sub-systems
        self._pid = PIDController()
        self._classifier = TerrainClassifier(
            terrain_size=zone_manager.terrain_size,
            cell_size=1.0,
        )
        self._scan_planner = ScanPlanner(
            terrain_size=zone_manager.terrain_size,
            altitude=scan_altitude,
            strip_spacing=SCAN_STRIP_SPACING,
        )

        # Selected landing zone (world x, y)
        self._target_zone = None        # set in ASSESS
        self._scan_retries = 0

        # State snapshot updated each tick
        self._state = None
        self._received_state = False

        print("[OODABackend] Initialised.")

    # ── Pegasus Backend interface ─────────────────────────────────────────────

    def initialize(self, vehicle):
        self._vehicle = vehicle
        print("[OODABackend] Vehicle registered.")

    def update_state(self, state):
        self._state = state
        self._received_state = True

    def update_sensor(self, sensor_type: str, data):
        # Pegasus IMU / GPS arrive here — not used for navigation in GPS-denied mode
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """LiDAR point cloud arrives here each tick during SCAN phase."""
        if sensor_type != "lidar_pointcloud":
            return
        if self._phase not in (Phase.SCAN, Phase.ASSESS):
            return
        if data is None:
            return

        # Pegasus delivers LiDAR as flat float32 array — reshape to (N, 4)
        try:
            pts_flat = np.array(data, dtype=np.float32)
            if pts_flat.ndim == 1 and len(pts_flat) % 4 == 0:
                pts = pts_flat.reshape(-1, 4)
            elif pts_flat.ndim == 2:
                pts = pts_flat
            else:
                return
            self._classifier.add_scan(pts[:, :3])
        except Exception as e:
            print(f"[OODABackend] LiDAR parse error: {e}")

    def input_reference(self):
        return self.input_ref

    def start(self):
        self._scan_planner.reset()
        print("[OODABackend] start()")

    def stop(self):
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self._phase = Phase.IDLE
        self._phase_timer = 0.0
        self._pid.reset()
        self._classifier.clear()
        self._scan_planner.reset()
        self.zone_manager.reset()
        self._target_zone = None
        self._scan_retries = 0
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        print("[OODABackend] reset()")

    # ── Main update tick ──────────────────────────────────────────────────────

    def update(self, dt: float):
        if not self._received_state or self._vehicle is None:
            return

        self._phase_timer += dt
        s = get_state(self._vehicle)

        if self._phase == Phase.IDLE:
            self._phase_idle(s)

        elif self._phase == Phase.TAKEOFF:
            self._phase_takeoff(s, dt)

        elif self._phase == Phase.SCAN:
            self._phase_scan(s, dt)

        elif self._phase == Phase.ASSESS:
            self._phase_assess(s)

        elif self._phase == Phase.APPROACH:
            self._phase_approach(s, dt)

        elif self._phase == Phase.LAND:
            self._phase_land(s, dt)

        elif self._phase == Phase.ABORT:
            self._phase_abort(s, dt)

    # ── Phase implementations ─────────────────────────────────────────────────

    def _phase_idle(self, s):
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self._pid.reset()
        self._scan_planner.reset()
        self._classifier.clear()
        self._phase = transition_phase(self._phase, Phase.TAKEOFF)
        self._phase_timer = 0.0

    def _phase_takeoff(self, s, dt):
        target = np.array([0.0, 0.0, self.scan_altitude])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)

        alt_err = abs(s["position"][2] - self.scan_altitude)
        speed   = np.linalg.norm(s["velocity"])
        if alt_err < TAKEOFF_TOLERANCE and speed < 0.5:
            self._phase = transition_phase(self._phase, Phase.SCAN)
            self._phase_timer = 0.0

    def _phase_scan(self, s, dt):
        # Drive to current scan waypoint
        if self._scan_planner.complete:
            # All waypoints visited — flush classifier and assess
            self._phase = transition_phase(self._phase, Phase.ASSESS)
            self._phase_timer = 0.0
            return

        wx, wy, wz = self._scan_planner.current_waypoint()
        target = np.array([wx, wy, wz])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)

        dist = np.linalg.norm(s["position"] - target)
        if dist < WAYPOINT_TOLERANCE:
            self._scan_planner.advance()

    def _phase_assess(self, s):
        """Run classifier on accumulated point cloud, update zone_manager, select target."""
        print(f"[OODA ASSESS] Coverage: {self.zone_manager.coverage_fraction():.1%}")

        # Push classifier results into zone_manager
        results = self._classifier.get_all_results()
        for (row, col), res in results.items():
            wx, wy = self._classifier.cell_to_world_centre(row, col)
            self.zone_manager.update_cell(wx, wy, res["p_safe"], res["variance"])

        best = self.zone_manager.best_zone()
        if best is not None:
            bx, by, bscore = best
            print(f"[OODA ASSESS] Best zone: ({bx:.1f}, {by:.1f}) score={bscore:.3f}")
            self._target_zone = (bx, by)
            self._phase = transition_phase(self._phase, Phase.APPROACH)
        else:
            print(f"[OODA ASSESS] No safe zone found. Retry {self._scan_retries+1}/{MAX_SCAN_RETRIES}")
            self._scan_retries += 1
            if self._scan_retries >= MAX_SCAN_RETRIES:
                print("[OODA ASSESS] Max retries reached — ABORT.")
                self._phase = transition_phase(self._phase, Phase.ABORT)
            else:
                # Expand scan and retry
                self._scan_planner.strip_spacing = max(
                    1.0, self._scan_planner.strip_spacing - 0.5
                )
                self._scan_planner.reset()
                self._classifier.clear()
                self._phase = transition_phase(self._phase, Phase.SCAN)
        self._phase_timer = 0.0

    def _phase_approach(self, s, dt):
        """Fly to APPROACH_ALTITUDE above the selected zone."""
        if self._target_zone is None:
            self._phase = transition_phase(self._phase, Phase.ABORT)
            return

        zx, zy = self._target_zone
        target = np.array([zx, zy, APPROACH_ALTITUDE + 0.0])  # zone z≈0
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)

        lateral_err = np.linalg.norm(s["position"][:2] - np.array([zx, zy]))
        alt_diff    = abs(s["position"][2] - APPROACH_ALTITUDE)
        if lateral_err < APPROACH_LAT_TOL and alt_diff < 0.5:
            self._phase = transition_phase(self._phase, Phase.LAND)
            self._phase_timer = 0.0
            # Import MPC here (Week 10 deliverable) — falls back to PID until implemented
            try:
                from controller.mpc_descent import MPCDescentOptimizer
                self._mpc = MPCDescentOptimizer()
                print("[OODABackend] MPC descent optimizer loaded.")
            except ImportError:
                self._mpc = None
                print("[OODABackend] MPC not available — using PID descent.")

    def _phase_land(self, s, dt):
        """Descend to touchdown. Uses MPC if available, else PID."""
        if self._target_zone is None:
            self._phase = transition_phase(self._phase, Phase.ABORT)
            return

        zx, zy = self._target_zone

        # Touchdown detection — LANDING SLIDE BUG FIX from dronos
        if s["position"][2] < LAND_ALT_ZERO:
            print("[OODA LAND] Touchdown detected.")
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            return

        if hasattr(self, "_mpc") and self._mpc is not None:
            # Week 10: MPC generates acceleration reference
            target_state = np.array([zx, zy, 0.0, 0.0, 0.0, 0.0])
            current_state = np.concatenate([s["position"], s["velocity"]])
            u_opt = self._mpc.compute(current_state, target_state)
            # Convert MPC acceleration to force/torque then rotor speeds
            mass = 1.50
            force = mass * (u_opt + np.array([0.0, 0.0, 9.81]))
            torque = np.zeros(3)
            self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)
        else:
            # Fallback: PID to ground directly below current position
            target = np.array([zx, zy, 0.0])
            force, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)

    def _phase_abort(self, s, dt):
        """No safe zone — return to origin and land."""
        print(f"[OODA ABORT] Returning home... (alt={s['position'][2]:.1f}m)")
        if s["position"][2] < LAND_ALT_ZERO:
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            return
        target = np.array([0.0, 0.0, max(0.0, s["position"][2] - 0.5)])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(self._vehicle, force, torque)

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def target_zone(self):
        return self._target_zone
