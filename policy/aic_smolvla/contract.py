"""Observation/action contract between AIC and SmolVLA.

Matches what AIC's LeRobot recorder (aic_utils/lerobot_robot_aic, cartesian mode) writes, so a
SmolVLA fine-tuned on CheatCode demos recorded with it can be served without changes:
  - observation.images.{left,center,right}_camera: RGB uint8, scaled by 0.25 (1152x1024 -> 288x256)
  - observation.state: 26 floats (TCP pose 7, TCP velocity 6, TCP error 6, joint positions 7)
  - action: 6 floats, a Cartesian twist (linear xyz m/s, angular xyz rad/s)

Pure numpy: imported by both the ROS policy (AIC pixi env) and the inference server (LeRobot venv).
"""

import numpy as np

CAMERAS = ("left_camera", "center_camera", "right_camera")
IMAGE_SCALE = 0.25

STATE_NAMES = (
    [f"tcp_pose.position.{a}" for a in "xyz"]
    + [f"tcp_pose.orientation.{a}" for a in "xyzw"]
    + [f"tcp_velocity.linear.{a}" for a in "xyz"]
    + [f"tcp_velocity.angular.{a}" for a in "xyz"]
    + [f"tcp_error.{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
    + [f"joint_positions.{i}" for i in range(7)]
)
ACTION_NAMES = [f"linear.{a}" for a in "xyz"] + [f"angular.{a}" for a in "xyz"]
STATE_DIM = len(STATE_NAMES)  # 26
ACTION_DIM = len(ACTION_NAMES)  # 6


def image_to_numpy(img_msg, scale: float = IMAGE_SCALE) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) RGB uint8, resized by `scale`."""
    channels = 4 if img_msg.encoding in ("rgba8", "bgra8") else 3
    img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.step // channels if img_msg.step else img_msg.width, channels
    )[:, : img_msg.width, :3]
    if img_msg.encoding.startswith("bgr"):
        img = img[..., ::-1]
    if scale != 1.0:
        import cv2

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def state_to_numpy(obs_msg) -> np.ndarray:
    """aic_model_interfaces/Observation -> (26,) float32 state, in STATE_NAMES order."""
    cs = obs_msg.controller_state
    p, q = cs.tcp_pose.position, cs.tcp_pose.orientation
    v, w = cs.tcp_velocity.linear, cs.tcp_velocity.angular
    joints = list(obs_msg.joint_states.position[:7])
    joints += [0.0] * (7 - len(joints))
    return np.array(
        [p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z, *cs.tcp_error, *joints],
        dtype=np.float32,
    )


def observation_to_numpy(obs_msg) -> dict:
    """Observation message -> plain dict of numpy arrays, ready to send to the server."""
    return {
        "state": state_to_numpy(obs_msg),
        "images": {
            cam: image_to_numpy(getattr(obs_msg, cam.replace("_camera", "_image")))
            for cam in CAMERAS
        },
    }


def task_prompt(task) -> str:
    """aic_task_interfaces/Task -> language instruction for SmolVLA."""
    return (
        f"insert the {task.plug_name or task.plug_type} plug into "
        f"{task.port_name or task.port_type} on {task.target_module_name}"
    )


def clip_twist(action: np.ndarray, max_linear: float, max_angular: float) -> np.ndarray:
    """Scale the linear and angular parts down so their norms stay within the limits."""
    a = np.asarray(action, dtype=np.float64)[:ACTION_DIM].copy()
    for sl, limit in ((slice(0, 3), max_linear), (slice(3, 6), max_angular)):
        n = np.linalg.norm(a[sl])
        if n > limit > 0:
            a[sl] *= limit / n
    return np.nan_to_num(a)
