# scanner/lidar_bridge.py
# Standalone ROS2 subscriber — runs in WSL2 system python3, NOT Isaac Sim python.
#
# Setup (run in every new terminal before launching this):
#   source /opt/ros/humble/setup.bash
#   export ROS_DOMAIN_ID=0
#   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
#   export FASTRTPS_DEFAULT_PROFILES_FILE=/mnt/d/project/terrascout/fastdds_wsl.xml
#
# Usage:
#   python3 scanner/lidar_bridge.py
#
# Writes each LiDAR frame to /tmp/terrascout_lidar.npy as float32 (N, 3) XYZ array.
# OODABackend reads this file each update tick when the ROS2 subscriber is unavailable.

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

OUTPUT_PATH = "/tmp/terrascout_lidar.npy"


class LidarBridge(Node):
    def __init__(self):
        super().__init__("lidar_bridge")
        self._frame_count = 0
        self.create_subscription(
            PointCloud2,
            "/drone0/sensors/lidar/points",
            self._callback,
            10,
        )
        self.get_logger().info(
            f"LidarBridge ready — subscribing to /drone0/sensors/lidar/points, "
            f"writing to {OUTPUT_PATH}"
        )

    def _callback(self, msg: PointCloud2):
        try:
            # Isaac Sim LiDAR publishes float32 fields x,y,z,intensity (point_step=16).
            # Decode binary buffer directly — no sensor_msgs_py dependency needed.
            pts = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(
                -1, msg.point_step // 4
            )[:, :3]  # drop intensity, keep XYZ
            np.save(OUTPUT_PATH, pts)
            self._frame_count += 1
            if self._frame_count % 100 == 0:
                self.get_logger().info(
                    f"Frame {self._frame_count}: {len(pts)} points saved to {OUTPUT_PATH}"
                )
        except Exception as e:
            self.get_logger().error(f"Decode error: {e}")


def main():
    rclpy.init()
    node = LidarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
