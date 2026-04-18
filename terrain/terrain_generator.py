# terrain/terrain_generator.py
# Procedural terrain scene builder for TerraScout.
# Replaces dronos scene/setup.py and box_manager.py.
#
# build_scene(app, pg) -> ZoneManager
#   Follows the same critical spawn order as dronos — do not reorder steps.

import numpy as np
import random

SEED = 42           # use fixed seed for graded demo reproducibility
TERRAIN_SIZE = 20   # terrain covers ±10 m in X and Y from world origin
SCAN_ALTITUDE = 30.0 # metres — TAKEOFF target and SCAN cruise altitude

# Blank ground-plane USD that ships with Isaac Sim (no warehouse geometry)
BLANK_ENV_USD = (
    "omniverse://localhost/NVIDIA/Assets/Isaac/5.1/Isaac/Environments/"
    "Simple_Room/simple_room.usd"
)
# If the above is unavailable, fall back to Pegasus built-in catalogue:
try:
    from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
    FALLBACK_ENV_USD = SIMULATION_ENVIRONMENTS["Rough Plane"]
except Exception:
    FALLBACK_ENV_USD = "/Isaac/Environments/Grid/default_environment.usd"


def build_scene(app, pg):
    """
    Build the TerraScout terrain scene.
    Returns a ZoneManager with the initial (all-UNKNOWN) grid.
    Caller (terrascout_main.py) calls timeline.play() after this returns.
    """
    from terrain.zone_manager import ZoneManager
    from controller.ooda_backend import OODABackend
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.params import ROBOTS
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    random.seed(SEED)
    np.random.seed(SEED)

    # ── e. Initialize world ──────────────────────────────────────────────────
    pg.initialize_world()

    # ── f. Load blank environment ────────────────────────────────────────────
    try:
        pg.load_environment(BLANK_ENV_USD)
    except Exception:
        pg.load_environment(FALLBACK_ENV_USD)

    # ── g. Buffer ─────────────────────────────────────────────────────────────
    for _ in range(10):
        app.update()

    # ── h. Create terrain prims ───────────────────────────────────────────────
    stage = omni.usd.get_context().get_stage()
    _create_terrain(stage)

    # ── i. ZoneManager + OODABackend + MultirotorConfig ───────────────────────
    zone_manager = ZoneManager(
        terrain_size=TERRAIN_SIZE,
        cell_size=1.0,
    )
    backend = OODABackend(zone_manager, scan_altitude=SCAN_ALTITUDE)
    config = MultirotorConfig()
    config.backends = [backend]

    # LiDAR graphical sensor — must be in config before Multirotor() so that
    # Lidar.initialize(vehicle) is called during spawn and creates the prim.
    #
    # Import directly: Lidar is commented out of graphical_sensors/__init__.py.
    #
    # Bug in Pegasus lidar.py line 68:
    #   config=self._sensor_configuration["sensor_configuration"]
    # self._sensor_configuration stores whatever config.get("sensor_configuration")
    # returns, then indexes it with ["sensor_configuration"].  Pass a dict so the
    # subscript resolves to the actual config string rather than crashing on a bare str.
    from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
    lidar = Lidar("lidar", config={
        "frequency":            10.0,
        "position":             np.array([0.0, 0.0, 0.10]),
        "orientation":          np.array([0.0, 0.0, 0.0]),
        "sensor_configuration": {"sensor_configuration": "Example_Rotary"},
        "show_render":          False,   # ROS2 publish handled by OmniGraph below
    })
    config.graphical_sensors = [lidar]

    # ── j. world.reset() ──────────────────────────────────────────────────────
    pg._world.reset()

    # ── k. Spawn drone ────────────────────────────────────────────────────────
    # Do NOT use pg.spawn_vehicle() — GUI-only path.
    # Multirotor() calls lidar.initialize(vehicle), which runs
    # IsaacSensorCreateRtxLidar and sets lidar._sensor to the created prim.
    Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [0.0, 0.0, 30.0], config=config)

    # ── l. OmniGraph action graphs ────────────────────────────────────────────
    _create_omnigraph(stage)

    # Wire the lidar prim to the OmniGraph ROS2RtxLidarHelper node via a
    # render product.  Must happen after both the Multirotor spawn (so the prim
    # exists) and the OmniGraph creation (so the LidarPub node exists).
    import omni.replicator.core as rep
    import omni.graph.core as og
    render_product = rep.create.render_product(lidar._sensor.GetPath(), [1, 1])
    lidar_pub = og.Controller.node("/World/TerrascoutGraph/LidarPub")
    og.Controller.attribute("inputs:renderProductPath", lidar_pub).set(
        render_product.path
    )

    # ── m. Final buffer ───────────────────────────────────────────────────────
    for _ in range(10):
        app.update()

    print(f"[TerrainGenerator] Scene built. "
          f"Zone grid: {zone_manager.grid_w}x{zone_manager.grid_h} cells.")
    return zone_manager


# ── Terrain geometry ──────────────────────────────────────────────────────────

def _create_terrain(stage):
    """Spawn boulders and slope ramps as USD prims with physics."""
    from pxr import UsdGeom, Gf, UsdPhysics

    terrain_root = stage.DefinePrim("/World/terrain", "Xform")

    # Boulders — random spheres, kept away from spawn point and centre pad
    n_boulders = random.randint(6, 10)
    for i in range(n_boulders):
        pos = _random_terrain_pos(min_r=2.0, max_r=9.0)
        r = random.uniform(0.3, 0.75)
        prim_path = f"/World/terrain/boulder_{i}"
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.GetRadiusAttr().Set(r)
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(r)))
        # Physics collider
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        print(f"[TerrainGenerator]   boulder_{i} at {pos}, r={r:.2f}")

    # Slope ramps — simple tilted flat meshes
    n_ramps = random.randint(2, 3)
    for i in range(n_ramps):
        pos = _random_terrain_pos(min_r=3.0, max_r=8.0)
        angle_deg = random.uniform(15, 25)
        _create_ramp(stage, f"/World/terrain/ramp_{i}", pos, angle_deg)
        print(f"[TerrainGenerator]   ramp_{i} at {pos}, slope={angle_deg:.1f}°")

    # Guaranteed flat landing pad (helps confirm classifier works)
    _create_flat_pad(stage, "/World/terrain/safe_pad", pos=[3.0, 4.0])
    print("[TerrainGenerator]   safe_pad at [3.0, 4.0]")


def _random_terrain_pos(min_r, max_r):
    """Uniform random position in annulus [min_r, max_r] from origin."""
    angle = random.uniform(0, 2 * np.pi)
    r = random.uniform(min_r, max_r)
    return [r * np.cos(angle), r * np.sin(angle)]


def _create_ramp(stage, prim_path, pos, angle_deg):
    """Create a tilted rectangular mesh as a slope ramp."""
    from pxr import UsdGeom, Gf, UsdPhysics
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    # Simple 2x3 m quad, points in local space
    w, l, h = 2.0, 3.0, 0.1
    pts = [(-w/2,-l/2,0),(w/2,-l/2,0),(w/2,l/2,0),(-w/2,l/2,0)]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in pts])
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    xf = UsdGeom.Xformable(mesh.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.0))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(angle_deg, 0.0, random.uniform(0, 360)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _create_flat_pad(stage, prim_path, pos):
    """Flat cylinder representing a guaranteed-safe landing pad."""
    from pxr import UsdGeom, Gf, UsdPhysics
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.GetRadiusAttr().Set(1.0)
    cyl.GetHeightAttr().Set(0.05)
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.025))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())


# ── OmniGraph ─────────────────────────────────────────────────────────────────

def _create_omnigraph(stage):
    """
    Create OmniGraph for odometry publish + LiDAR publish.
    Node types verified against Isaac Sim 5.1.0 — do not change type strings.
    """
    import omni.graph.core as og

    keys = og.Controller.Keys
    (graph, _, _, _) = og.Controller.edit(
        {"graph_path": "/World/TerrascoutGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick",       "omni.graph.action.OnPlaybackTick"),
                ("SimTime",      "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Context",  "isaacsim.ros2.bridge.ROS2Context"),
                ("ClockPub",     "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("OdomComp",     "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("OdomPub",      "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("TFPub",        "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("LidarPub",     "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick",          "ClockPub.inputs:execIn"),
                ("OnTick.outputs:tick",           "OdomComp.inputs:execIn"),
                ("OnTick.outputs:tick",           "TFPub.inputs:execIn"),
                ("SimTime.outputs:simulationTime","ClockPub.inputs:timeStamp"),
                ("OdomComp.outputs:execOut",      "OdomPub.inputs:execIn"),
                ("ROS2Context.outputs:context",   "ClockPub.inputs:context"),
                ("ROS2Context.outputs:context",   "OdomPub.inputs:context"),
                ("ROS2Context.outputs:context",   "TFPub.inputs:context"),
                ("ROS2Context.outputs:context",   "LidarPub.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("OdomComp.inputs:chassisPrim",   ["/World/quadrotor"]),
                ("LidarPub.inputs:topicName",     "/drone0/sensors/lidar/points"),
                ("LidarPub.inputs:frameId",       "lidar"),
            ],
        },
    )
    print("[TerrainGenerator] OmniGraph created.")
