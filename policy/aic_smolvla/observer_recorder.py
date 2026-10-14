"""Save third-person observer camera frames (from scripts/observer_cameras.py) to disk.

    PYTHONPATH=policy pixi run python -m aic_smolvla.observer_recorder OUT_DIR [view ...]

Writes OUT_DIR/<view>/<sim_time_ns>.jpg for every frame of /observer/<view>/image, so
scripts/make_video.py can match them to recorded episodes by simulation time.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ObserverRecorder(Node):
    def __init__(self, out: Path, views):
        super().__init__("observer_recorder")
        self.counts = {}
        for view in views:
            (out / view).mkdir(parents=True, exist_ok=True)
            self.counts[view] = 0
            self.create_subscription(Image, f"/observer/{view}/image",
                                     lambda msg, v=view: self.save(out / v, v, msg), qos_profile_sensor_data)
        self.get_logger().info(f"recording observer views {views} -> {out}")

    def save(self, folder, view, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        if msg.encoding.startswith("rgb"):
            img = img[..., ::-1]
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        cv2.imwrite(str(folder / f"{stamp:019d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.counts[view] += 1
        if self.counts[view] % 100 == 0:
            self.get_logger().info(f"{view}: {self.counts[view]} frames")


def main():
    out = Path(sys.argv[1]).expanduser()
    views = sys.argv[2:] or ["overview", "side"]
    rclpy.init()
    node = ObserverRecorder(out, views)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f"saved {node.counts}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
