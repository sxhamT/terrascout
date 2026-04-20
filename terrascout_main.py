# terrascout_main.py
# Entry point for TerraScout simulation.
# Mirrors dronos launch_gui.py — spawn order MUST NOT change.
# Run via launch_ros2.cmd (plain cmd, no admin).

import sys
import argparse as _ap
import numpy as np

# Parse --seed before any Isaac Sim imports.  parse_known_args ignores
# Omniverse's own flags (--/rtx/..., etc.) so there is no conflict.
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument('--seed', type=int, default=42,
                     help='Terrain seed (default 42 — graded demo)')
_sim_args, _ = _parser.parse_known_args()
_sim_seed = _sim_args.seed
del _ap, _parser, _sim_args
print(f"[TerraScout] Seed={_sim_seed}")

# ── 1. SimulationApp (full kit) ──────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ── 2. Enable ROS2 bridge before any Pegasus import ─────────────────────────
import carb
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

# ── 3. Buffer — let bridge finish loading ────────────────────────────────────
for _ in range(10):
    simulation_app.update()

# ── 4. Pegasus interface ─────────────────────────────────────────────────────
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
pg = PegasusInterface()

# ── 5. Build scene (terrain, drone, backends) ────────────────────────────────
from terrain.terrain_generator import build_scene
zone_manager, lidar_prim_path, _ooda_backend = build_scene(simulation_app, pg, seed=_sim_seed)

# ── 6. OmniGraph (clock + odometry only) ────────────────────────────────────
# LiDAR data is provided by synthetic analytical raycast inside OODABackend.
# No render product, no annotator, no ROS2RtxLidarHelper needed.
import omni.graph.core as og
import omni.usd

from terrain.terrain_generator import _create_omnigraph
_stage = omni.usd.get_context().get_stage()
_create_omnigraph(_stage, og)

for _ in range(5):
    simulation_app.update()

# ── 7. Play ──────────────────────────────────────────────────────────────────
import omni.timeline
timeline = omni.timeline.get_timeline_interface()
timeline.play()

for _ in range(10):
    simulation_app.update()

# ── 7b. Overview camera ───────────────────────────────────────────────────────
from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(
    eye=np.array([0.0, -30.0, 30.0]),
    target=np.array([0.0, 0.0, 0.0]),
)

# ── 8. LiDAR zone map visualiser (floating omni.ui window) ─────────────────
from controller.lidar_viz import LidarViz
_viz = LidarViz(
    terrain_size=zone_manager.terrain_size,
    cell_size=zone_manager.cell_size,
    px=420,
    update_every=15,   # refresh every 15 sim ticks (~4 Hz at 60 Hz sim)
)

print("[TerraScout] Simulation running. Ctrl+C or close window to stop.")

# ── 9. Main loop ─────────────────────────────────────────────────────────────
try:
    while simulation_app.is_running():
        simulation_app.update()
        _viz.refresh(zone_manager, _ooda_backend)
except KeyboardInterrupt:
    print("[TerraScout] Interrupted.")
finally:
    _viz.close()
    timeline.stop()
    simulation_app.close()
