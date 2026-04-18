# terrascout_main.py
# Entry point for TerraScout simulation.
# Mirrors dronos launch_gui.py — spawn order MUST NOT change.
# Run via launch_ros2.cmd (plain cmd, no admin).

import sys

# ── 1. SimulationApp (full kit) ──────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

# ── 2. Enable ROS2 bridge before any Pegasus import ─────────────────────────
import carb
from isaacsim.core.utils.extensions import set_extension_enabled_immediate
set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# ── 3. Buffer — let bridge finish loading ────────────────────────────────────
for _ in range(10):
    simulation_app.update()

# ── 4. Pegasus interface ─────────────────────────────────────────────────────
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
pg = PegasusInterface()

# ── 5. Build scene (terrain, drone, backends) ────────────────────────────────
from terrain.terrain_generator import build_scene
zone_manager = build_scene(simulation_app, pg)

# ── 6. Play ──────────────────────────────────────────────────────────────────
import omni.timeline
timeline = omni.timeline.get_timeline_interface()
timeline.play()

print("[TerraScout] Simulation running. Ctrl+C or close window to stop.")

# ── 7. Main loop ─────────────────────────────────────────────────────────────
try:
    while simulation_app.is_running():
        simulation_app.update()
except KeyboardInterrupt:
    print("[TerraScout] Interrupted.")
finally:
    timeline.stop()
    simulation_app.close()
