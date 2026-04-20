# CLAUDE.md — TerraScout
# For: Claude Code (WSL2) — primary interface for all code changes
# Last updated: April 20, 2026 (Day 7 — sim stabilisation + viz)

---

## How to Use This File (Claude Code Instructions)

You are Claude Code running in WSL2. The project lives at:
- Windows: `D:\project\terrascout`
- WSL2:    `/mnt/d/project/terrascout`

**All Python files run on Windows under the Isaac Sim interpreter.**
Never add Linux shebangs, Linux absolute paths, or `#!/usr/bin/env python`.
Never run Python files yourself — output the file, the human runs it.

When making changes:
1. Read the file first before editing
2. Make the smallest correct change
3. State clearly what you changed and why
4. If a change touches the spawn order, stop and warn explicitly

---

## One-Line Problem Statement

> "Can autonomous drones reliably detect and select safe landing zones in
> GPS-denied unstructured terrain, where rotor-strike and tip-over failures
> are mission-terminal?"

---

## Course Context (AI Analytics — CMU 12-831)

**Report format:** ACM BuildSyS sigconf LaTeX template

**Rubric:**
- Mastery of concepts (35%) <- most important
- Formatting and organization (15%)
- Grammar, clarity, accuracy (15%)
- Pros/cons of solution (15%)
- Creativity (10%)
- Overall idea and execution (10%)

**Module coverage map — what this project must demonstrate:**

| Module | Topic | How TerraScout covers it |
|--------|-------|--------------------------|
| M3 | Measurement uncertainty | Kalman variance gates classification confidence |
| M4 | Predictive modeling | MPC descent trajectory prediction |
| M5 | Monte Carlo | 50 terrain variants = MC validation |
| M6 | Frequentist / MLE | Gaussian likelihood functions in Bayesian classifier |
| M7 | Bayesian modeling + AI agents | Bayesian terrain classifier + OODA loop |
| M8 | Classification | Terrain safety classification from LiDAR |
| M9 | OODA + PID + Kalman | All three — required by professor explicitly |
| M10 | MPC | Descent trajectory optimizer |
| M11 | Middleware + integration | ROS2 bridge + Isaac Sim / WSL2 |

**Critical:** Pure geometric classification (PCA slope only) is NOT sufficient.
The Bayesian + Kalman components are required for the mastery grade.

---

## What TerraScout Is

Autonomous terrain-scanning quadrotor digital twin in Isaac Sim 5.1.0.
A single quadrotor OODA loop:
1. Takes off to 15 m scan altitude
2. Executes lawnmower LiDAR sweep over procedural terrain
3. Classifies each cell: geometric pre-filter -> Kalman height tracker -> Bayesian P(safe)
4. Selects best zone where P(safe) > 0.85 AND Kalman variance < threshold
5. Flies MPC-optimised descent — target updates as estimate refines mid-descent
6. Touches down — no GPS at any point

**Why MPC not just PID for descent:** Zone estimate refines continuously as
drone descends and LiDAR resolution improves. MPC receding horizon re-plans
each tick as the estimate updates. PID needs a fixed target. This is the core
technical justification — not accuracy, but online replanning under moving estimate.

**Why failure is mission-terminal:** GPS-denied = no manual recovery.
Tip-over or rotor-strike on touchdown ends the mission permanently.
Small drones are MORE susceptible to tip-over (low righting moment).

**Physics framing:** GPS-denied terrestrial (mountainous SAR, disaster response,
RF-jammed). Dragonfly/Titan cited in future work only — do not claim
interplanetary in main body (Iris quadrotor physics invalid on Mars/Moon).

---

## Stack

| Item | Value |
|------|-------|
| OS | Windows 11 |
| GPU | RTX 3060 12GB |
| Driver | 580.97 Game Ready — PINNED, do not upgrade |
| Isaac Sim | 5.1.0 at D:\isaacsim-env |
| Kit config | isaacsim.exp.full.kit |
| Pegasus | v5.1.0 editable at D:\PegasusSimulator\extensions\pegasus.simulator |
| ROS2 | Humble — bundled DLLs inside Isaac Sim, NOT system ROS2 |
| Python | 3.11 at D:\isaacsim-env\Scripts\python.exe |
| Project | D:\project\terrascout (WSL2: /mnt/d/project/terrascout) |
| Dronos ref | D:\project\dronos — complete working reference, do not modify |

---

## Repository Structure (all files exist unless marked)

```
D:\project\terrascout\
|-- CLAUDE.md
|-- terrascout_main.py                 # LidarViz integrated in main loop
|-- launch_ros2.cmd
|-- fastdds_wsl.xml                    # auto-generated, do not edit
|-- .gitignore
|
|-- config/
|   |-- topics.yaml
|   +-- mpc_params.yaml
|
|-- controller/
|   |-- __init__.py
|   |-- act.py                         # from dronos — do not modify
|   |-- observe.py                     # from dronos — do not modify
|   |-- pid_controller.py              # from dronos — do not modify
|   |-- decide.py
|   |-- ooda_backend.py                # cell_size=0.5, step=0.25, _last_scan_pts, min_radius=0.7
|   |-- ooda_backend_old.py            # original backup — do not use
|   |-- terrain_classifier.py          # exists — Kalman+Bayesian classifier
|   |-- lidar_viz.py                   # live top-down zone map (omni.ui window)
|   +-- mpc_descent.py
|
|-- terrain/
|   |-- __init__.py
|   |-- terrain_generator.py           # smart bowl placement, sigma=2.0, slope colors, cell_size=0.5
|   |-- terrain_generator_old.py       # original backup — do not use
|   +-- zone_manager.py                # all fixes applied — cluster-based zone selection
|
|-- scanner/
|   |-- __init__.py
|   |-- lidar_classifier.py            # correct, do not modify
|   +-- scan_planner.py                # correct
|
+-- eval/
    |-- __init__.py
    |-- synthetic_terrain.py
    |-- run_eval.py
    |-- mpc_vs_pid_benchmark.py
    +-- plot_results.py
```

---

## CURRENT STATUS: Day 7 — Sim stabilised, ready for eval + report

Cluster detection bug (no safe zones after 500 scans) was root-caused and fixed.
Cause: previous terrain_generator placed Gaussian bowls at random positions that
landed on steep wave crests. New version uses `_pick_bowl_positions()` which
pre-screens the terrain grid and only places bowls where background slope < 12°.

Live zone map visualiser (LidarViz) added — floating omni.ui window shows
grey/green/yellow/red grid, cyan scan footprint, white cross on committed zone.

**Immediate actions:**
1. Run sim once to confirm TOUCHDOWN now works with new terrain
2. Run `python eval/run_eval.py --n 50 --seed 42` for MC results
3. Run `python eval/mpc_vs_pid_benchmark.py --n 20 --seed 42`
4. Run `python eval/plot_results.py` for LaTeX-ready figures
5. Write report following the outline below

**terrain_classifier.py** — DONE (Day 2). Implements Kalman + Bayesian + geometric
pre-filter as required by M6/M7/M9. Do not modify unless a bug surfaces in eval.

**Key tuned parameters (as of Day 7):**
- `SCAN_ALTITUDE = 15.0` m
- `cell_size = 0.5` m (ZoneManager + TerrainClassifier — finer grid than original 1.0)
- `min_radius_m = 0.7` for cluster detection
- LiDAR raycast `step = 0.25` m (~5000 pts/scan at 15m alt)
- Bowl sigma = 2.0 m (~8–10m effective landing zone diameter)
- Bowl depth = adaptive: `min(wave_z - 0.1, 0.8)` m

---

## Pending Fixes — ALL DONE as of Day 6

### Fix 1 — ooda_backend.py: PID call signature (BREAKS AT RUNTIME)

pid_controller.py actual signature:
```python
def compute(self, position, velocity, orientation_quat, angular_velocity, target_position, dt)
```

Current wrong calls in ooda_backend.py (4 places):
```python
force, torque = self._pid.compute(s, target, dt)   # WRONG
```

Correct:
```python
force, torque = self._pid.compute(
    s["position"], s["velocity"], s["orientation"],
    s["angular_velocity"], target, dt
)
```

Also fix constructor (PIDController takes no arguments):
```python
self._pid = PIDController(mass=1.50)  ->  self._pid = PIDController()
```

Affected methods: _phase_takeoff, _phase_scan, _phase_approach, _phase_abort

### Fix 2 — ooda_backend.py: Wrong altitude constants

```python
SCAN_ALTITUDE = 8.0     ->  SCAN_ALTITUDE = 15.0   # NOTE: settled on 15m not 30m for sim
APPROACH_ALTITUDE = 3.0 ->  approach_z = target_ground_z + 8.0  # 8m AGL
```

### Fix 3 — ooda_backend.py: Swap LidarClassifier for TerrainClassifier

```python
# Remove these imports:
from scanner.lidar_classifier import LidarClassifier
self._classifier = LidarClassifier(cell_size=1.0, terrain_size=...)

# Add:
from controller.terrain_classifier import TerrainClassifier
self._classifier = TerrainClassifier(terrain_size=zone_manager.terrain_size, cell_size=1.0)
```

In _phase_assess, replace score push loop:
```python
# Remove:
scores = self._classifier.compute_scores()
for (row, col), score in scores.items():
    wx, wy = self._classifier.cell_to_world_centre(row, col)
    self.zone_manager.update_cell(wx, wy, score)

# Add:
results = self._classifier.get_all_results()
for (row, col), res in results.items():
    wx, wy = self._classifier.cell_to_world_centre(row, col)
    self.zone_manager.update_cell(wx, wy, res["p_safe"], res["variance"])
```

In update_graphical_sensor, replace:
```python
self._classifier.add_points(pts[:, :3])  ->  self._classifier.add_scan(pts[:, :3])
```

### Fix 4 — terrain_generator.py: Spawn altitude + environment fallback

```python
# Spawn height: drone spawns low (3.0m), TAKEOFF phase climbs to SCAN_ALTITUDE
[0.0, 0.0, 0.5]  ->  [0.0, 0.0, 3.0]

# Environment: replaced hardcoded USD path with:
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
pg.load_environment(SIMULATION_ENVIRONMENTS["Rough Plane"])
```

### Fix 5 — zone_manager.py: Add variance field + update signature

Add to __init__:
```python
self._variances = np.full((n, n), np.nan, dtype=np.float32)
```

Change update_cell signature:
```python
def update_cell(self, wx: float, wy: float, p_safe: float, variance: float = 0.0):
```

Update SAFE threshold logic:
```python
# SAFE only when both conditions met:
if p_safe >= SAFE_THRESHOLD and variance < 0.05:
    self._status[row, col] = ZoneStatus.SAFE
```

Add to reset():
```python
self._variances[:] = np.nan
```

---

## Critical Spawn Order (DO NOT CHANGE — any deviation crashes)

```
terrascout_main.py:
  a. SimulationApp full kit
  b. set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
  c. app.update() x10
  d. PegasusInterface()

terrain/terrain_generator.py  build_scene(app, pg):
  e. pg.initialize_world()
  f. pg.load_environment(env_usd)
  g. app.update() x10
  h. _create_terrain(stage)
  i. ZoneManager + OODABackend + MultirotorConfig
  j. pg._world.reset()
  k. Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [0,0,3.0], config=config)
  l. _create_omnigraph(stage)
  m. app.update() x10

terrascout_main.py:
  n. timeline.play()
  o. while app.is_running(): app.update()
```

---

## Critical Known Issues

1. OODABackend must NOT call super().__init__()
   Set self._vehicle = None directly.

2. force_and_torques_to_velocities() returns values, does NOT apply them.
   Must store in self.input_ref, returned by input_reference() each tick.

3. pg.initialize_world() BEFORE pg.load_environment() — always.

4. Always buffer app.update() x10 between major init steps.

5. NEVER press Update in Extension Manager for pegasus.simulator.
   Fix: rmdir /s /q C:\Users\cardi\AppData\Local\ov\data\exts\v2\pegasus.simulator
   Verify: Script Editor -> from pegasus.simulator.params import ROOT; print(ROOT)
   Must print D:\PegasusSimulator\extensions not AppData.

6. LiDAR data shape: arrives as flat float32 (N*4,).
   Reshape: pts = data.reshape(-1, 4)  cols=[x,y,z,intensity]
   Filter:  pts = pts[pts[:,2] > -0.1]

7. LiDAR OmniGraph: use ROS2RtxLidarHelper, not deprecated token node.
   Arrives in update_graphical_sensor("lidar_pointcloud", data).

8. Landing slide bug: zero input_ref when state["position"][2] < 0.15 in LAND.

9. ~~PID signature mismatch in ooda_backend.py~~ — FIXED (Day 6). All 4 call sites updated.

10. terrain_generator.py FLAT_PADS format: uses {"cx","cy","depth","sigma"} — NOT {"cx","cy","r"}.
    ooda_backend.py _synthetic_lidar_scan() must use the Gaussian formula to match _heightmap_z().
    The two files are tightly coupled — never update one without checking the other.

11. _old backup files exist for ooda_backend and terrain_generator.
    To revert: cp controller/ooda_backend_old.py controller/ooda_backend.py
               cp terrain/terrain_generator_old.py terrain/terrain_generator.py

---

## PID Controller Reference

```python
# Correct call:
force, torque = pid.compute(
    s["position"],         # (3,) ENU
    s["velocity"],         # (3,) ENU
    s["orientation"],      # (4,) [qx,qy,qz,qw]
    s["angular_velocity"], # (3,) FLU body
    target,                # (3,) desired position
    dt                     # float
)
# Returns (scalar_thrust_N, torque_vec_3)
# Then: input_ref = vehicle.force_and_torques_to_velocities(force, torque)
```

Gains: Kp=diag(10,10,10), Kd=diag(8.5,8.5,8.5), Ki=diag(1.5,1.5,1.5),
       Kr=diag(3.5,3.5,3.5), Kw=diag(0.5,0.5,0.5), mass=1.50 kg

---

## OODA Phases

```
IDLE -> TAKEOFF -> DESCEND_SCAN -> APPROACH -> LAND
                        |
                        v (no safe zone at world z=8m floor)
                      ABORT
```

| Phase | Exit condition |
|-------|---------------|
| IDLE | immediate |
| TAKEOFF | alt_err < 0.30m AND speed < 0.5 m/s (target: SCAN_ALTITUDE=15m) |
| DESCEND_SCAN | best cluster found AND lidar_frames >= 3, OR forced at world z=8m |
| APPROACH | lateral_err < 0.5m AND 6m < alt_AGL < 10m (hovers at 8m AGL) |
| LAND | alt_AGL < 0.05m (TOUCHDOWN) |
| ABORT | triggered from DESCEND_SCAN when floor reached with no zone |

Phase enum in controller/decide.py — import from there only.

---

## Offline Eval (no Isaac Sim needed)

```bash
# From /mnt/d/project/terrascout in WSL2
# pip install numpy scipy matplotlib

python eval/run_eval.py --n 50 --seed 42
# -> eval/results/summary.json, per_variant.csv, summary.txt

python eval/mpc_vs_pid_benchmark.py --n 20 --seed 42
# -> eval/results/mpc_vs_pid.json, mpc_vs_pid.txt

python eval/plot_results.py
# -> eval/figures/*.pdf  (ACM sigconf 3.33in, LaTeX-ready)
```

Run this BEFORE worrying about sim. Numbers are independent of demo.
The 50 variants are a TEST SET — no training, no trainable parameters.

---

## OmniGraph Nodes (verified Isaac Sim 5.1.0)

```
omni.graph.action.OnPlaybackTick
isaacsim.core.nodes.IsaacComputeOdometry
isaacsim.core.nodes.IsaacReadSimulationTime
isaacsim.ros2.bridge.ROS2Context
isaacsim.ros2.bridge.ROS2PublishClock
isaacsim.ros2.bridge.ROS2PublishOdometry
isaacsim.ros2.bridge.ROS2PublishTransformTree
isaacsim.ros2.bridge.ROS2RtxLidarHelper
```

---

## ROS2 Topics

| Topic | Type | Source |
|-------|------|--------|
| /odom | nav_msgs/Odometry | OmniGraph |
| /clock | rosgraph_msgs/Clock | OmniGraph |
| /tf | tf2_msgs/TFMessage | OmniGraph |
| /drone0/state/pose | geometry_msgs/PoseStamped | Pegasus |
| /drone0/state/twist | geometry_msgs/TwistStamped | Pegasus |
| /drone0/sensors/imu | sensor_msgs/Imu | Pegasus |
| /drone0/sensors/lidar/points | sensor_msgs/PointCloud2 | OmniGraph |
| /terrascout/zone_map | visualization_msgs/MarkerArray | OODABackend |
| /terrascout/phase | std_msgs/String | OODABackend |

GPS intentionally absent — GPS-denied scenario.

WSL2 monitor setup (every new terminal):
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/mnt/d/project/terrascout/fastdds_wsl.xml
ros2 topic list
```

---

## Report Outline (ACM BuildSyS sigconf)

1. Abstract
2. Introduction — GPS-denied problem, mission-terminal framing
3. Related Work — TRN, HDA, Ingenuity, Dragonfly
4. System Architecture — digital twin closed loop
5. Terrain Classifier — Kalman + Bayesian + geometric pre-filter
6. OODA Controller — phase machine, confidence thresholds
7. MPC Descent — J formulation, constraints, receding horizon motivation
8. Synthetic Validation — Monte Carlo, 50 variants, ground truth method
9. Results — precision/recall, MPC vs PID, figures
10. Discussion — pros/cons, honest limitations
11. Future Work — real hardware, Dragonfly/Titan
12. References

Honest limitations for pros/cons:
- Synthetic clouds cleaner than real LiDAR (no beam divergence, multipath)
- Bayesian likelihood params manually specified, not empirically fitted
- MPC vs PID advantage modest on short descent — benefit is constraint
  enforcement and replanning under moving estimate, not raw accuracy

---

## Git

```
main                          <- stable only
feature/scaffold              [merged]
feature/terrain-classifier    [merged]
feature/ooda-fixes            [merged]
feature/lidar-setup           [ACTIVE] — synthetic raycast, terrain tuning, verbose OODA logging,
                                          descent speed fixes, MPC lateral-only LAND,
                                          smart bowl placement, denser scan grid (step=0.25),
                                          LidarViz live zone map, cell_size=0.5 throughout
```

```cmd
git checkout main
git checkout -b feature/name
git add . && git commit -m "feat: description"
git push -u origin feature/name
git checkout main && git merge feature/name && git push
```

---

## Sprint Progress

| Day | Status | Summary |
|-----|--------|---------|
| Day 1 | done | Scaffold, all files placed, folders corrected |
| Day 2 | done | terrain_classifier.py (Kalman+Bayesian), all 5 fixes applied, eval pipeline runs |
| Day 3 | done | Full OODA end-to-end in sim — synthetic LiDAR raycast (RTX annotator crashes on Win), DESCEND_SCAN -> LAND -> TOUCHDOWN verified |
| Day 4 | done | Terrain roughness tuned (high-freq heightmap), height-based vertex colors, boulder registry for raycast, ZoneManager margin fix |
| Day 5 | done | Verbose phase logging (DESCEND_SCAN 2s, zone-selected banner, LAND per-tick, TOUCHDOWN/ABORT banners), fastdds_wsl.xml gitignored |
| Day 6 | done | Descent speed fixes: _descent_target_z (0.02m/tick + vz hold), _land_target_z (0.01m/tick + vz freeze), MPC lateral-only in LAND (u[2]=0), velocity cap helper _vel_cap_force(2.0 m/s) applied in all non-LAND phases |
| Day 7 | done | Root-caused no-cluster bug (random bowl placement on steep terrain). Fixed with _pick_bowl_positions() selecting slope<12° candidates. sigma=2.0, cell_size=0.5, step=0.25. Added LidarViz live zone map. |
| Day 8-10 | in progress | Report — eval pipeline first, then ACM LaTeX writeup |

---

## Update This File

End of each session: check off sprint days, add new known issues,
update What Works. Mark fixed items in Pending Fixes as DONE.
