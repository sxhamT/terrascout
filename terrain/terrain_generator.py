# terrain/terrain_generator.py
# Procedural terrain scene builder for TerraScout.
# Replaces dronos scene/setup.py and box_manager.py.
#
# build_scene(app, pg) -> ZoneManager
#   Follows the same critical spawn order as dronos — do not reorder steps.

import numpy as np
import random

from isaacsim.core.utils.prims import set_targets

SEED = 42            # use fixed seed for graded demo reproducibility
TERRAIN_SIZE = 10    # terrain covers ±10 m in X and Y (20 x 20 m total)
SCAN_ALTITUDE = 15.0 # metres — TAKEOFF target altitude

# Module-level boulder registry — populated by _create_terrain().
# Used by _boulder_height_at() for synthetic raycast in OODABackend.
BOULDER_LIST = []   # each entry: {"cx": float, "cy": float, "r": float, "z": float}


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

    # ── f. Load environment ──────────────────────────────────────────────────
    from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
    pg.load_environment(SIMULATION_ENVIRONMENTS["Rough Plane"])

    # ── g. Buffer ─────────────────────────────────────────────────────────────
    for _ in range(10):
        app.update()

    # Hide the base environment (Rough Plane grid and any other USD prims loaded
    # by load_environment).  Our own heightmap fills the visible area.
    # Called before _create_terrain so our prims are added after and remain visible.
    stage = omni.usd.get_context().get_stage()
    _hide_environment_prims(stage)

    # ── h. Create terrain prims ───────────────────────────────────────────────
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
        "show_render":          False,   # data read via annotator in terrascout_main.py
    })
    config.graphical_sensors = [lidar]

    # ── j. world.reset() ──────────────────────────────────────────────────────
    pg._world.reset()

    # ── k. Spawn drone ────────────────────────────────────────────────────────
    # Do NOT use pg.spawn_vehicle() — GUI-only path.
    # Multirotor() calls lidar.initialize(vehicle), which runs
    # IsaacSensorCreateRtxLidar and sets lidar._sensor to the created prim.
    Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [0.0, 0.0, 30.0], config=config)

    # Confirm lidar prim was created by Multirotor init.
    # If this prints None, Pegasus failed to create the RTX lidar prim.
    print("[LiDAR] lidar._sensor after Multirotor spawn:", lidar._sensor)

    # ── Bottom-facing camera (drone-body frame) ───────────────────────────────
    # Slightly behind and above centre, tilted 15 deg forward so it shows
    # terrain ahead of the drone rather than straight down.
    camera_path = "/World/quadrotor/body/bottom_cam"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.GetFocalLengthAttr().Set(18.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 200.0))
    xf_cam = UsdGeom.Xformable(camera.GetPrim())
    xf_cam.AddTranslateOp().Set(Gf.Vec3f(0.0, -1.0, -0.3))
    xf_cam.AddRotateXYZOp().Set(Gf.Vec3f(-75.0, 0.0, 0.0))
    print("[Camera] Bottom camera attached at", camera_path)

    # ── Overview camera (fixed world position) ────────────────────────────────
    # Diagonal top-down view from (15, -15, 20) looking at terrain centre.
    # Switch to this in the viewport to watch the full descent.
    ov_path = "/World/overview_cam"
    ov_cam = UsdGeom.Camera.Define(stage, ov_path)
    ov_cam.GetFocalLengthAttr().Set(24.0)
    ov_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 500.0))
    xf_ov = UsdGeom.Xformable(ov_cam.GetPrim())
    xf_ov.AddTranslateOp().Set(Gf.Vec3f(15.0, -15.0, 20.0))
    xf_ov.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))
    print("[Camera] Overview camera at /World/overview_cam")

    # ── l. OmniGraph — created by caller AFTER render product exists ─────────
    # _create_omnigraph() is intentionally NOT called here.
    # terrascout_main.py calls it after rep.create.render_product() so that
    # LidarPub sees a valid renderProductPath from its very first tick.

    # Capture the lidar prim path for the caller to wire after timeline.play().
    # Render product creation MUST happen after play() — doing it here produces
    # a dead render product that never receives RTX data (0 Hz on the topic).
    lidar_prim_path = str(lidar._sensor.GetPath())
    print("[LiDAR] prim path (wire after play):", lidar_prim_path)

    # ── m. Final buffer ───────────────────────────────────────────────────────
    for _ in range(10):
        app.update()

    print(f"[TerrainGenerator] Scene built. "
          f"Zone grid: {zone_manager.grid_w}x{zone_manager.grid_h} cells.")
    return zone_manager, lidar_prim_path, backend


# ── Environment cleanup ───────────────────────────────────────────────────────

def _hide_environment_prims(stage):
    """
    Hide every prim that already exists under /World right after load_environment().
    Our terrain, lights, and quadrotor are added after this call so they stay visible.
    """
    from pxr import UsdGeom
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        return
    hidden = []
    for child in world.GetChildren():
        imageable = UsdGeom.Imageable(child)
        if imageable:
            imageable.MakeInvisible()
            hidden.append(str(child.GetPath()))
    if hidden:
        print(f"[TerrainGenerator] Hidden environment prims: {hidden}")


# ── Terrain geometry ──────────────────────────────────────────────────────────

def _create_terrain(stage):
    """Spawn boulders and slope ramps as USD prims with physics."""
    from pxr import UsdGeom, Gf, UsdPhysics, UsdLux, Semantics, Sdf, Vt

    terrain_root = stage.DefinePrim("/World/terrain", "Xform")

    # ── Lighting ──────────────────────────────────────────────────────────────
    # DomeLight: fills sky ambient so nothing appears black
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.GetIntensityAttr().Set(1000.0)
    dome.GetColorAttr().Set(Gf.Vec3f(0.8, 0.9, 1.0))  # slight sky blue tint

    # DistantLight: directional sun — angled 45° to cast visible shadows
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/SunLight")
    sun.GetIntensityAttr().Set(2500.0)
    sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.85))  # warm sunlight
    sun.GetAngleAttr().Set(0.53)
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))
    print("[TerrainGenerator] Lights created.")

    # ── Heightmap mesh — continuous undulating rocky surface ─────────────────
    _create_heightmap(stage, "/World/terrain/ground")
    print("[TerrainGenerator]   heightmap mesh created.")

    # ── Boulders on top of the heightmap — hard collision obstacles ───────────
    # Radii 0.30–0.80 m: proportionate to 0.5 m drone footprint.
    # Placement within ±8.5 m so boulders land on the heightmap mesh (±10 m extent).
    # Each boulder is also registered in BOULDER_LIST for synthetic raycast.
    BOULDER_LIST.clear()
    n_boulders = random.randint(8, 12)
    for i in range(n_boulders):
        pos = _random_terrain_pos(min_r=1.5, max_r=8.5)
        r = random.uniform(0.30, 0.80)
        hz = _heightmap_z(pos[0], pos[1])
        prim_path = f"/World/terrain/boulder_{i}"
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.GetRadiusAttr().Set(r)
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(hz + r)))
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdGeom.Imageable(sphere.GetPrim()).MakeVisible()
        # Rocky dark-grey boulder colour
        bpv = UsdGeom.PrimvarsAPI(sphere.GetPrim()).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.constant,
        )
        bpv.Set(Vt.Vec3fArray([Gf.Vec3f(0.15, 0.12, 0.10)]))
        sem = Semantics.SemanticsAPI.Apply(sphere.GetPrim(), "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set("terrain")
        BOULDER_LIST.append({"cx": pos[0], "cy": pos[1], "r": r, "z": hz})
        print(f"[TerrainGenerator]   boulder_{i} at ({pos[0]:.1f},{pos[1]:.1f}), r={r:.2f}")

    # Guaranteed flat landing pad — cleared area so classifier finds a SAFE zone.
    # Placed at a fixed offset from origin; z anchored to the heightmap surface.
    pad_x, pad_y = 3.0, 4.0
    pad_z = _heightmap_z(pad_x, pad_y)
    _create_flat_pad(stage, "/World/terrain/safe_pad", pos=[pad_x, pad_y], z=pad_z)
    print(f"[TerrainGenerator]   safe_pad at ({pad_x},{pad_y}), z={pad_z:.2f}")


def _random_terrain_pos(min_r, max_r):
    """Uniform random position in annulus [min_r, max_r] from origin."""
    angle = random.uniform(0, 2 * np.pi)
    r = random.uniform(min_r, max_r)
    return [r * np.cos(angle), r * np.sin(angle)]


def _heightmap_z(x: float, y: float) -> float:
    """
    Evaluate the procedural heightmap at world position (x, y).
    Uses overlapping sine/cosine waves — no external libraries needed.

    Frequencies chosen so that slope varies meaningfully across 1 m cells:
      Layer 1 (f=1.05): wavelength ~6 m, max slope ~24°
      Layer 2 (f=1.57): wavelength ~4 m, max slope ~26°
      Layer 3 (f=3.14): wavelength ~2 m, max slope ~32°  (hits pre-filter)
    This forces the Bayesian classifier to discriminate — the old low-frequency
    terrain (~35 m wavelength) made every 1 m cell trivially flat.
    """
    h  =  0.40 * np.sin(1.05 * x + 0.9) * np.cos(0.90 * y + 0.4)
    h +=  0.30 * np.cos(1.57 * x - 0.5) * np.sin(1.40 * y + 1.2)
    h +=  0.20 * np.sin(3.14 * x + 1.7) * np.cos(2.80 * y - 0.8)
    # Bias upward so surface sits above z=0
    h += 0.40
    return float(np.clip(h, -0.5, 1.5))


def _boulder_height_at(wx: float, wy: float) -> float:
    """
    Height contribution from boulders at world position (wx, wy).
    Returns the maximum spherical cap height from any overlapping boulder.
    Used by the synthetic raycast to make boulders appear in the point cloud.
    """
    z = 0.0
    for b in BOULDER_LIST:
        dist = np.sqrt((wx - b["cx"]) ** 2 + (wy - b["cy"]) ** 2)
        if dist < b["r"]:
            cap = 1.0 - (dist / b["r"]) ** 2
            z = max(z, b["r"] * np.sqrt(max(cap, 0.0)))
    return z


def _create_heightmap(stage, prim_path: str, grid: int = 80):
    """
    Build a USD Mesh heightmap over the full TERRAIN_SIZE x TERRAIN_SIZE area.

    Parameters
    ----------
    grid : int
        Number of vertices per side.  80 → 79×79 quads over 40 m = 0.5 m resolution.
    """
    from pxr import UsdGeom, Gf, UsdPhysics, Vt, Semantics, Sdf

    extent = TERRAIN_SIZE                   # half-size: vertices span [-extent, extent]
    xs = np.linspace(-extent, extent, grid)
    ys = np.linspace(-extent, extent, grid)

    # Build vertex array (grid*grid, 3)
    verts = []
    heights = []
    for iy in range(grid):
        for ix in range(grid):
            z = _heightmap_z(xs[ix], ys[iy])
            verts.append(Gf.Vec3f(float(xs[ix]), float(ys[iy]), float(z)))
            heights.append(z)

    # Build quad faces: each quad = 2 triangles → use faceVertexCounts = [4]*n_quads
    face_counts = []
    face_indices = []
    for iy in range(grid - 1):
        for ix in range(grid - 1):
            i00 = iy * grid + ix
            i10 = iy * grid + ix + 1
            i01 = (iy + 1) * grid + ix
            i11 = (iy + 1) * grid + ix + 1
            face_counts.append(4)
            face_indices.extend([i00, i10, i11, i01])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(verts))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh.GetSubdivisionSchemeAttr().Set("none")

    # doubleSided = True: RTX ray-traces both faces.
    # The quad winding [i00,i10,i11,i01] is CW from above → normals point DOWN.
    # Without doubleSided the LiDAR (above, looking down) hits the back face and
    # gets no return.  This is the primary reason RTX LiDAR misses the terrain.
    mesh.GetDoubleSidedAttr().Set(True)

    # Height-based vertex colours — high-contrast ramp with power-curve stretch.
    heights_np = np.array(heights, dtype=np.float32)
    h_min, h_max = heights_np.min(), heights_np.max()
    h_norm = (heights_np - h_min) / (h_max - h_min + 1e-6)
    h_norm = np.power(h_norm, 0.5)   # sqrt stretches low values for more contrast
    colors = []
    for h in h_norm:
        if h < 0.5:
            # Dark rocky brown lowlands
            t = h / 0.5
            r, g, b = 0.35 + t * 0.20, 0.25 + t * 0.15, 0.15 + t * 0.10
        elif h < 0.75:
            # Mid grey rock
            t = (h - 0.5) / 0.25
            r, g, b = 0.55 + t * 0.20, 0.52 + t * 0.20, 0.50 + t * 0.20
        else:
            # Dramatic white peaks
            t = (h - 0.75) / 0.25
            r, g, b = 0.75 + t * 0.25, 0.75 + t * 0.25, 0.75 + t * 0.25
        colors.append(Gf.Vec3f(float(min(r, 1.0)), float(min(g, 1.0)), float(min(b, 1.0))))

    primvar_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    color_pv = primvar_api.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    color_pv.Set(Vt.Vec3fArray(colors))

    mesh_prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh_prim)

    # Ensure the mesh is visible in the render layer and raycasted by RTX LiDAR.
    UsdGeom.Imageable(mesh_prim).MakeVisible()
    sem = Semantics.SemanticsAPI.Apply(mesh_prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set("terrain")


def _create_flat_pad(stage, prim_path, pos, z: float = 0.0):
    """Flat cylinder representing a guaranteed-safe landing pad."""
    from pxr import UsdGeom, Gf, UsdPhysics, Semantics
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.GetRadiusAttr().Set(1.0)
    cyl.GetHeightAttr().Set(0.05)
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(z + 0.025)))
    pad_prim = cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pad_prim)
    UsdGeom.Imageable(pad_prim).MakeVisible()
    sem = Semantics.SemanticsAPI.Apply(pad_prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set("terrain")


# ── OmniGraph ─────────────────────────────────────────────────────────────────

def _create_omnigraph(stage, og):
    """
    Create OmniGraph for clock and odometry ROS2 publish only.
    LiDAR data is provided by synthetic analytical raycast in OODABackend —
    no render product, no ROS2RtxLidarHelper, no annotator API needed.
    """
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/World/TerrascoutGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SimTime",        "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Context",    "isaacsim.ros2.bridge.ROS2Context"),
                ("ClockPub",       "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("OdomComp",       "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("OdomPub",        "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick",          "ClockPub.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",          "OdomComp.inputs:execIn"),
                ("SimTime.outputs:simulationTime",       "ClockPub.inputs:timeStamp"),
                ("OdomComp.outputs:execOut",             "OdomPub.inputs:execIn"),
                ("OdomComp.outputs:position",            "OdomPub.inputs:position"),
                ("OdomComp.outputs:orientation",         "OdomPub.inputs:orientation"),
                ("OdomComp.outputs:linearVelocity",      "OdomPub.inputs:linearVelocity"),
                ("OdomComp.outputs:angularVelocity",     "OdomPub.inputs:angularVelocity"),
                ("ROS2Context.outputs:context",          "ClockPub.inputs:context"),
                ("ROS2Context.outputs:context",          "OdomPub.inputs:context"),
            ],
            keys.SET_VALUES: [],
        },
    )

    # chassisPrim is a USD relationship — cannot be set via SET_VALUES.
    # Must use set_targets() exactly as dronos does.
    from isaacsim.core.utils import stage as stage_utils
    current_stage = stage_utils.get_current_stage()
    set_targets(
        prim=current_stage.GetPrimAtPath("/World/TerrascoutGraph/OdomComp"),
        attribute="inputs:chassisPrim",
        target_prim_paths=["/World/quadrotor"],
    )
    print("[TerrainGenerator] OmniGraph created (clock + odometry).")
