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
DESCEND_RATE           = 0.1    # m/s — slow descent gives scanner time to build density
DESCEND_ABORT_ALT      = 1.5    # m — below this with no safe zone -> ABORT
LAND_ALT_ZERO          = 0.05   # metres — below this -> touchdown detected
LAND_ZONE_MARGIN       = 2.0    # m — reject zones within this dist of terrain edge

LIDAR_SCAN_EVERY_N     = 3      # call synthetic scan every N update ticks
DESCEND_MIN_SCANS      = 15     # minimum LiDAR frames before committing to a zone


class OODABackend(Backend):
    """
    Autonomous terrain-scanning OODA loop for TerraScout.

    Phase machine: IDLE -> TAKEOFF -> DESCEND_SCAN -> APPROACH -> LAND -> ABORT

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
            cell_size=0.5,
        )
        self._mpc = None  # loaded lazily at TAKEOFF exit

        # Selected landing zone (world x, y) and terrain height — committed in DESCEND_SCAN
        self._target_zone     = None
        self._target_ground_z = 0.0   # actual terrain height at landing zone

        # State snapshot updated each tick
        self._state = None
        self._received_state = False

        # LiDAR frame counters
        self._lidar_frame_count = 0   # synthetic scan frames fed to classifier

        # Descent target z trackers — smoothly stepped each tick
        self._descent_target_z = float(scan_altitude)  # steps 0.02m/tick in DESCEND_SCAN
        self._land_target_z    = None                  # initialised at LAND entry, 0.01m/tick

        self._last_scan_pts = None   # last synthetic scan — for top-down display
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
        self._target_zone     = None
        self._target_ground_z = 0.0
        self._lidar_frame_count = 0
        self._descent_target_z = float(self.scan_altitude)
        self._land_target_z    = None
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self._last_scan_pts = None
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
        step = 0.25

        xs = np.arange(drone_pos[0] - fov_radius,
                       drone_pos[0] + fov_radius + step, step, dtype=np.float32)
        ys = np.arange(drone_pos[1] - fov_radius,
                       drone_pos[1] + fov_radius + step, step, dtype=np.float32)

        if len(xs) == 0 or len(ys) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        gx, gy = np.meshgrid(xs, ys)
        wx = gx.ravel()
        wy = gy.ravel()

        # Vectorised heightmap — must match _heightmap_z() exactly
        from terrain.terrain_generator import FLAT_PADS
        wz = (0.80 * np.sin(1.05 * wx + 0.9) * np.cos(0.90 * wy + 0.4)
            + 0.60 * np.cos(1.57 * wx - 0.5) * np.sin(1.40 * wy + 1.2)
            + 0.20 * np.sin(3.14 * wx + 1.7) * np.cos(2.80 * wy - 0.8)
            + 0.10 * np.cos(4.71 * wx + 0.3) * np.sin(4.71 * wy - 1.1)
            + 0.40)
        # Gaussian bowl depressions — must match _heightmap_z() exactly
        for pad in FLAT_PADS:
            cx, cy = pad["cx"], pad["cy"]
            depth, sigma = pad["depth"], pad["sigma"]
            dist2 = (wx - cx) ** 2 + (wy - cy) ** 2
            wz = wz - depth * np.exp(-dist2 / (2.0 * sigma ** 2))
        # Perimeter wall — matches _heightmap_z() rim term
        rim = np.maximum(np.maximum(np.abs(wx), np.abs(wy)) - 7.0, 0.0)
        wz = wz + 3.0 * rim * rim
        wz = np.clip(wz, -0.5, 20.0)

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

        pts_out = np.column_stack([wx, wy, wz]).astype(np.float32)
        self._last_scan_pts = pts_out   # stored for top-down visualiser
        return pts_out

    def _compute_ground_z(self, wx: float, wy: float) -> float:
        """
        Compute actual terrain surface height at world position (wx, wy).
        Matches the synthetic raycast: heightmap + max boulder spherical cap.
        Used so all altitude logic is terrain-relative (AGL), not world-absolute.
        """
        from terrain.terrain_generator import _heightmap_z, BOULDER_LIST
        gz = _heightmap_z(float(wx), float(wy))
        boulder_z = 0.0
        for b in BOULDER_LIST:
            dist = float(np.sqrt((wx - b["cx"]) ** 2 + (wy - b["cy"]) ** 2))
            if dist < b["r"]:
                cap = max(0.0, 1.0 - (dist / b["r"]) ** 2)
                boulder_z = max(boulder_z, b["r"] * float(np.sqrt(cap)))
        return gz + boulder_z

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
        self._classifier.clear()
        self._phase = transition_phase(self._phase, Phase.TAKEOFF)
        self._phase_timer = 0.0

    def _phase_takeoff(self, s, dt):
        target = np.array([0.0, 0.0, self.scan_altitude])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(
            self._vehicle, self._vel_cap_force(float(force), s), torque)

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
        Slow descent from SCAN_ALTITUDE to world z=8.0m floor.
        Continuously course-corrects laterally toward the best zone estimate
        as it updates (SpaceX-style convergence visible in sim).
        Pure PID actuation — MPC is reserved for LAND phase only.
        Commits to zone when found + min scans met, or forced at 8m floor.
        """
        alt = float(s["position"][2])

        # Periodic status print every 2 seconds.
        # _compute_ground_z is called only here (not every tick) to avoid overhead.
        if int(self._phase_timer / 2.0) > int((self._phase_timer - dt) / 2.0):
            drone_gz = self._compute_ground_z(float(s["position"][0]), float(s["position"][1]))
            best_p = self.zone_manager.best_landing_zone(min_radius_m=0.7, margin=LAND_ZONE_MARGIN)
            best_str = (f"P={best_p[2]:.3f} at ({best_p[0]:.1f},{best_p[1]:.1f})"
                        if best_p is not None else "none")
            print(f"[DESCEND_SCAN] alt={alt:.1f}m  alt_agl={alt - drone_gz:.1f}m  "
                  f"scans={self._lidar_frame_count}  "
                  f"coverage={self.zone_manager.coverage_fraction():.1%}  "
                  f"safe_cells={self.zone_manager.safe_zone_count()}  "
                  f"best={best_str}")

        # Push classifier results into zone_manager only on scan ticks — no new
        # data arrives between scans so pushing every tick is pure overhead.
        if self._tick_count % LIDAR_SCAN_EVERY_N == 0:
            results = self._classifier.get_all_results()
            for (row, col), res in results.items():
                wx, wy = self._classifier.cell_to_world_centre(row, col)
                self.zone_manager.update_cell(wx, wy, res["p_safe"], res["variance"])

        best = self.zone_manager.best_landing_zone(min_radius_m=0.7, margin=LAND_ZONE_MARGIN)
        if best is None:
            print(f"[DESCEND_SCAN] no valid cluster  scans={self._lidar_frame_count}"
                  f"  safe_cells={self.zone_manager.safe_zone_count()}"
                  f"  coverage={self.zone_manager.coverage_fraction():.1%}")

        # Zone commit: normal (min scans met AND drone is near-hover AND has descended)
        # OR forced at floor.
        # Speed gate: don't commit while speed > 0.8 m/s.
        # Descent gate: require _descent_target_z has stepped at least 3 m below scan
        # altitude — prevents committing at scan altitude immediately after TAKEOFF
        # stabilises, which would hand APPROACH a 10 m dive.
        speed = float(np.linalg.norm(s["velocity"]))
        normal_commit = (best is not None
                         and self._lidar_frame_count >= DESCEND_MIN_SCANS
                         and speed < 0.8
                         and self._descent_target_z <= self.scan_altitude - 3.0)
        forced_commit = best is not None and alt <= 8.0

        if normal_commit or forced_commit:
            bx, by, bscore = best
            cell = self.zone_manager.world_to_cell(bx, by)
            var = float(self.zone_manager._variances[cell[0], cell[1]]) if cell is not None else float("nan")
            self._target_ground_z = self._compute_ground_z(bx, by)
            # Safety: if _compute_ground_z returns <=0 (heightmap edge case or
            # amplitude changes making the formula return near-zero), sample a
            # small neighbourhood and take the max so AGL math is never corrupted.
            if self._target_ground_z <= 0.0:
                offsets = [(0,0),(0.5,0),(-0.5,0),(0,0.5),(0,-0.5)]
                samples = [self._compute_ground_z(bx + dx, by + dy) for dx, dy in offsets]
                self._target_ground_z = max(samples)
                print(f"[WARN] ground_z was <=0 at ({bx:.1f},{by:.1f}) — "
                      f"neighbourhood max = {self._target_ground_z:.3f}m")
            clusters = self.zone_manager.get_all_clusters(min_radius_m=0.7, margin=LAND_ZONE_MARGIN)
            print(f"[ASSESS] Valid landing clusters (>=1.5m radius): {len(clusters)}")
            for cx_s, cy_s, sc, sz in clusters[:5]:
                print(f"         ({cx_s:.1f},{cy_s:.1f}) score={sc:.3f} size={sz} cells")
            pos = s["position"]
            print(f"\n{'=' * 50}")
            print(f"[OODA] ZONE SELECTED: ({bx:.1f}, {by:.1f})")
            print(f"       P(safe)={bscore:.3f}  Kalman_var={var:.4f}")
            print(f"       Ground elevation: {self._target_ground_z:.2f}m")
            print(f"       Drone at ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
            if forced_commit and not normal_commit:
                print(f"       [FORCED: world z=8m floor, scans={self._lidar_frame_count}]")
            print(f"{'=' * 50}\n")
            self._target_zone = (bx, by)
            self._phase = transition_phase(self._phase, Phase.APPROACH)
            self._phase_timer = 0.0
            return

        # Abort: hit world z=8m floor with no zone found
        if alt <= 8.0:
            cov = self.zone_manager.coverage_fraction()
            n_safe = self.zone_manager.safe_zone_count()
            print(f"\n{'!' * 50}")
            print(f"[ABORT] No safe zone found (alt={alt:.1f}m, floor=8.0m)")
            print(f"  Coverage: {cov:.1%}  Safe zones: {n_safe}")
            print(f"  Returning to origin")
            print(f"{'!' * 50}\n")
            self._phase = transition_phase(self._phase, Phase.ABORT)
            self._phase_timer = 0.0
            return

        # Lateral course-correction toward best zone, pure PID.
        # If a zone is found but speed gate not yet met, hold current altitude to brake
        # before committing — prevents dive overshoot into APPROACH.
        # _descent_target_z steps at most 0.02m per tick, held when vz < -1.0 m/s.
        tx, ty = (best[0], best[1]) if best is not None else (0.0, 0.0)
        vz = float(s["velocity"][2])
        zone_found_braking = (best is not None
                              and self._lidar_frame_count >= DESCEND_MIN_SCANS
                              and speed >= 0.8)
        if zone_found_braking:
            # Snap _descent_target_z to the actual approach altitude (ground_z + 8.0)
            # so APPROACH inherits a drone already at its hover target — no climb.
            estimated_gz = self._compute_ground_z(float(best[0]), float(best[1]))
            self._descent_target_z = max(self._descent_target_z, estimated_gz + 8.0)
            target_z = self._descent_target_z
        elif vz > -1.0:
            self._descent_target_z = max(8.0, self._descent_target_z - 0.02)
            target_z = self._descent_target_z
        else:
            target_z = self._descent_target_z
        target = np.array([tx, ty, target_z])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(
            self._vehicle, self._vel_cap_force(float(force), s), torque)

    def _phase_approach(self, s, dt):
        """
        PID transit to 8 m AGL directly above the committed landing zone.
        zone_found_braking snaps _descent_target_z to ground_z+8.0 before commit,
        so the drone arrives here already at approach altitude — no climb needed.
        """
        if self._target_zone is None:
            self._phase = transition_phase(self._phase, Phase.ABORT)
            return

        zx, zy = self._target_zone
        approach_z = self._target_ground_z + 8.0   # 8m AGL above zone
        alt = float(s["position"][2])
        alt_above_zone = alt - self._target_ground_z
        lat_err = float(np.sqrt((s["position"][0] - zx) ** 2 + (s["position"][1] - zy) ** 2))
        speed = float(np.linalg.norm(s["velocity"]))

        # Progress print every 2 seconds
        if int(self._phase_timer / 2.0) > int((self._phase_timer - dt) / 2.0):
            print(f"[APPROACH] alt={alt:.1f}m  alt_above_zone={alt_above_zone:.1f}m  "
                  f"lateral_err={lat_err:.2f}m  speed={speed:.2f}m/s  "
                  f"target=({zx:.1f},{zy:.1f},{approach_z:.1f})")

        # Transition to LAND once laterally aligned at 8m AGL.
        # Altitude band 6-10m above zone prevents early transition during descent.
        if lat_err < 0.5 and 6.0 < alt_above_zone < 10.0:
            print(f"[APPROACH] Aligned above zone (lat_err={lat_err:.2f}m, AGL={alt_above_zone:.1f}m) -> LAND")
            self._phase = transition_phase(self._phase, Phase.LAND)
            self._phase_timer = 0.0
            return

        target = np.array([zx, zy, approach_z])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(
            self._vehicle, self._vel_cap_force(float(force), s), torque)

    def _phase_land(self, s, dt):
        """
        MPC final descent from 5 m AGL to touchdown.
        All z targets use self._target_ground_z — never hardcoded 0.
        Touchdown detection uses AGL, not world-z.
        """
        if self._target_zone is None:
            self._phase = transition_phase(self._phase, Phase.ABORT)
            return

        zx, zy = self._target_zone
        gz = self._target_ground_z

        alt = float(s["position"][2])
        alt_agl = alt - gz
        lat_err = float(np.sqrt((s["position"][0] - zx) ** 2 + (s["position"][1] - zy) ** 2))
        vz = float(s["velocity"][2])
        lateral_mode = "MPC-lateral+PID-z" if self._mpc is not None else "PID-descent"
        land_z_str = f"{self._land_target_z:.2f}" if self._land_target_z is not None else "init"
        # Print every 1 second only — was printing every tick (60Hz spam)
        if int(self._phase_timer / 1.0) > int((self._phase_timer - dt) / 1.0):
            print(f"[LAND] alt_agl={alt_agl:.2f}m  lat_err={lat_err:.2f}m  "
                  f"vz={vz:.2f}m/s  tgt_z={land_z_str}  mode={lateral_mode}")

        if alt_agl < LAND_ALT_ZERO:
            fx, fy, fz = float(s["position"][0]), float(s["position"][1]), float(s["position"][2])
            final_lat = float(np.sqrt((fx - zx) ** 2 + (fy - zy) ** 2))
            print(f"\n{'*' * 50}")
            print(f"[TOUCHDOWN] Mission complete!")
            print(f"  Landing zone: ({zx:.1f}, {zy:.1f})  ground_z={gz:.2f}m")
            print(f"  Final position: ({fx:.1f}, {fy:.1f}, {fz:.2f})")
            print(f"  Lateral error: {final_lat:.3f}m")
            print(f"  Total LiDAR scans: {self._lidar_frame_count}")
            print(f"{'*' * 50}\n")
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            return

        # _land_target_z initialised once at LAND entry, then steps 0.05m/tick downward.
        # 0.01m/tick was too slow at sim rates — drone hovered indefinitely.
        if self._land_target_z is None:
            self._land_target_z = float(s["position"][2])
        self._land_target_z = max(gz, self._land_target_z - 0.05)

        # Velocity cap: if descending faster than 1.0 m/s, freeze target to brake.
        # Threshold tightened from 1.5 to 1.0 for safer final approach.
        vz = float(s["velocity"][2])
        if vz < -1.0:
            self._land_target_z = float(s["position"][2])  # freeze target, let drone catch up

        if self._mpc is not None:
            # MPC handles lateral correction only (xy).
            # Vertical z controlled by PID with slow-step _land_target_z.
            target_state = np.array([zx, zy, self._land_target_z, 0.0, 0.0, 0.0])
            current_state = np.concatenate([s["position"], s["velocity"]])
            u_opt = self._mpc.compute(current_state, target_state)
            u_opt[2] = 0.0   # zero MPC vertical — PID handles z

            # MPC xy correction: offset the PID target by the MPC acceleration
            # scaled by dt (not dt*dt — that was negligibly small and caused hover freeze).
            # Clamp to ±0.3m so MPC doesn't fight PID aggressively at close range.
            mpc_xy_offset = np.array([float(u_opt[0]), float(u_opt[1]), 0.0]) * dt
            target_mpc = np.array([zx, zy, self._land_target_z])
            target_mpc[:2] += np.clip(mpc_xy_offset[:2], -0.3, 0.3)
            force, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target_mpc, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, float(force), torque)
        else:
            target = np.array([zx, zy, self._land_target_z])
            force, torque = self._pid.compute(
                s["position"], s["velocity"], s["orientation"],
                s["angular_velocity"], target, dt
            )
            self.input_ref = compute_rotor_velocities(self._vehicle, float(force), torque)

    def _phase_abort(self, s, dt):
        """No safe zone — descend to origin using AGL touchdown detection."""
        origin_gz = self._compute_ground_z(0.0, 0.0)
        alt_agl = float(s["position"][2]) - origin_gz
        if alt_agl < LAND_ALT_ZERO:
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            return
        target = np.array([0.0, 0.0, max(origin_gz, s["position"][2] - 0.5)])
        force, torque = self._pid.compute(
            s["position"], s["velocity"], s["orientation"],
            s["angular_velocity"], target, dt
        )
        self.input_ref = compute_rotor_velocities(
            self._vehicle, self._vel_cap_force(float(force), s), torque)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vel_cap_force(self, force: float, s, max_speed: float = 2.0) -> float:
        """
        Scale down thrust when speed exceeds max_speed (m/s).
        Equivalent to the Fix-4 velocity cap requested for pid_controller.py,
        applied here so the PID source file (from dronos) stays unmodified.
        """
        speed = float(np.linalg.norm(s["velocity"]))
        if speed > max_speed:
            force = force * (max_speed / speed)
        return force

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def target_zone(self):
        return self._target_zone
