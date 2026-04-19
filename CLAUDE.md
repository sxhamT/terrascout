# CLAUDE.md — TerraScout
# For: Claude Code (WSL2) — primary interface for all code changes
# Last updated: April 19, 2026

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
1. Takes off to 30 m scan altitude
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
|-- terrascout_main.py
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
|   |-- ooda_backend.py                # HAS BUGS — see Pending Fixes
|   |-- terrain_classifier.py          # DOES NOT EXIST YET — write first
|   +-- mpc_descent.py
|
|-- terrain/
|   |-- __init__.py
|   |-- terrain_generator.py           # HAS BUGS — see Pending Fixes
|   +-- zone_manager.py                # NEEDS UPDATE — see Pending Fixes
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

## IMMEDIATE NEXT TASK: Write terrain_classifier.py

File: `controller/terrain_classifier.py`

This is the primary AI component required by the course (M6/M7/M9).
It wraps `lidar_classifier.py` (which already exists and must not be changed).

**Pipeline:**

```
Raw LiDAR points
    |
    v
lidar_classifier.py (existing, unchanged)
    outputs per cell: slope_deg, roughness_rms, clearance_m
    |
    v
terrain_classifier.py (new)
    |
    +-- Stage 1: Geometric pre-filter (O(N), fast)
    |     slope > 30 deg -> UNSAFE immediately, skip Bayesian
    |
    +-- Stage 2: Kalman filter per cell (M9)
    |     State:   estimated mean height z_bar
    |     Predict: z_bar stable (static terrain), P grows by Q=0.001
    |     Update:  K = P / (P + R),  R = 0.02^2 (LiDAR noise)
    |              z_bar = z_bar + K*(measurement - z_bar)
    |              P = (1-K)*P
    |     Gate:    only classify when P < 0.05 m^2
    |
    +-- Stage 3: Bayesian posterior per cell (M6/M7)
          Prior:      P(safe) = 0.5
          Likelihood: P(slope | safe)   = Gaussian(mean=0,  std=5 deg)
                      P(slope | unsafe) = Gaussian(mean=20, std=8 deg)
                      P(rough | safe)   = Gaussian(mean=0,  std=0.05)
                      P(rough | unsafe) = Gaussian(mean=0.2,std=0.08)
          Update:     log posterior = log prior + log likelihood (numerical stability)
          Output:     P(safe) in [0,1], updated each scan pass
```

**Classification thresholds:**
- P(safe) > 0.85 AND variance < 0.05 -> SAFE
- P(safe) < 0.20                      -> UNSAFE
- Otherwise                           -> UNCERTAIN

**Required interface:**
```python
class TerrainClassifier:
    def __init__(self, terrain_size: float, cell_size: float)
    def add_scan(self, points: np.ndarray)        # called each LiDAR frame
    def get_cell_result(self, row, col) -> dict   # {"p_safe", "variance", "status", "n_obs"}
    def get_all_results(self) -> dict             # {(row,col): result_dict}
    def cell_to_world_centre(self, row, col)      # -> (wx, wy)
    def clear(self)                               # reset for new scan pass
    def kalman_converged(self, row, col) -> bool  # variance < 0.05
```

Do NOT modify lidar_classifier.py. TerrainClassifier calls it internally.

---

## Pending Fixes (apply after terrain_classifier.py exists)

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
SCAN_ALTITUDE = 8.0     ->  SCAN_ALTITUDE = 30.0
APPROACH_ALTITUDE = 3.0 ->  APPROACH_ALTITUDE = 5.0
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
# Spawn height:
[0.0, 0.0, 0.5]  ->  [0.0, 0.0, 30.0]

# Environment fallback (replace hardcoded USD path):
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
FALLBACK_ENV_USD = SIMULATION_ENVIRONMENTS["Rough Plane"]
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
  k. Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [0,0,30.0], config=config)
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

9. PID signature mismatch in ooda_backend.py — see Fix 1 above. NOT YET FIXED.

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
IDLE -> TAKEOFF -> SCAN -> ASSESS -> APPROACH -> LAND
                    ^____________|  (retry, max 2x)
ABORT (max retries exceeded)
```

| Phase | Exit condition |
|-------|---------------|
| IDLE | immediate |
| TAKEOFF | alt >= 28m AND speed < 0.5 m/s |
| SCAN | scan_planner.complete |
| ASSESS | best P(safe) > 0.85 AND variance < 0.05 |
| APPROACH | lateral_err < 0.5m AND alt within 0.5m of 5m |
| LAND | z < 0.05m |

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
feature/terrain-classifier    <- write terrain_classifier.py here
feature/ooda-fixes            <- apply Fixes 1-5 after classifier done
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
| Day 6-10 | - | Report |

---

## Update This File

End of each session: check off sprint days, add new known issues,
update What Works. Mark fixed items in Pending Fixes as DONE.
