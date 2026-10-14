#!/usr/bin/env python3
"""Add third-person "observer" cameras to the running AIC Gazebo world and bridge them to ROS.

    scripts/observer_cameras.py            # run on the host while the simulator is up

Spawns static camera models (no collision geometry, so they can't touch anything or affect scoring)
into world `aic_world` inside the aic_eval container, then runs ros_gz_bridge so the images appear as
ROS topics /observer/<view>/image. Record them with policy/aic_smolvla/observer_recorder.py.
Runs until interrupted.
"""

import math
import os
import subprocess
import sys
import time

# name: (camera position, look-at point), world frame. Robot base is at (-0.2, 0.2, 1.14),
# the task board around (0.15..0.2, -0.2..0.05, 1.14).
VIEWS = {
    "overview": ((0.95, 0.80, 1.85), (0.0, -0.05, 1.22)),
    "side": ((-0.05, -1.25, 1.50), (0.0, -0.05, 1.22)),
}
WIDTH, HEIGHT, FOV, RATE = 960, 540, 1.05, 10


def camera_sdf(name, eye, target):
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))  # positive pitch looks down in Gazebo
    return f"""<?xml version="1.0"?>
<sdf version="1.9"><model name="observer_{name}"><static>true</static>
  <pose>{eye[0]} {eye[1]} {eye[2]} 0 {pitch:.4f} {yaw:.4f}</pose>
  <link name="link"><sensor name="camera" type="camera">
    <camera><horizontal_fov>{FOV}</horizontal_fov>
      <image><width>{WIDTH}</width><height>{HEIGHT}</height><format>R8G8B8</format></image>
      <clip><near>0.05</near><far>20</far></clip></camera>
    <always_on>1</always_on><update_rate>{RATE}</update_rate>
    <topic>/observer/{name}/image</topic>
  </sensor></link></model></sdf>"""


def in_container(cmd, **kw):
    uid = str(os.getuid())
    env = "source /ws_aic/install/setup.bash && export RMW_IMPLEMENTATION=rmw_zenoh_cpp && "
    return subprocess.Popen(["docker", "exec", "-u", uid, "aic_eval", "bash", "-c", env + cmd], **kw)


def main():
    for name, (eye, target) in VIEWS.items():
        sdf = camera_sdf(name, eye, target).replace("'", "'\\''")
        for attempt in range(12):  # the world may still be loading
            p = in_container(f"ros2 run ros_gz_sim create -world aic_world -name observer_{name} -string '{sdf}'")
            if p.wait(timeout=60) == 0:
                break
            time.sleep(5)
        else:
            sys.exit(f"failed to spawn observer_{name}")
        print(f"spawned observer_{name} at {eye} looking at {target}", flush=True)

    topics = " ".join(f"/observer/{n}/image@sensor_msgs/msg/Image[gz.msgs.Image" for n in VIEWS)
    bridge = in_container(f"exec ros2 run ros_gz_bridge parameter_bridge {topics}")
    print(f"bridging {list(VIEWS)} -> /observer/<view>/image", flush=True)
    try:
        while bridge.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.terminate()
        in_container("pkill -f 'parameter_bridge /observer' || true").wait()


if __name__ == "__main__":
    main()
