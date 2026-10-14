"""Record AIC episodes (any policy) to disk while the policy runs.

One folder per episode under the record root:
    <root>/<YYYYmmdd_HHMMSS>_<policy>_<port>/
        meta.json                    task, prompt, policy, frame count, insertion result
        data.npz                     state (N, 26), sim_t (N,), tcp_velocity (N, 6)
        left_camera/000000.jpg ...   RGB frames, 288x256, JPEG q95 (also center_camera/, right_camera/)
        preview.mp4                  three cameras side by side, for viewing

Frames are sampled at `hz` (sim time) from the same Observation the policy sees, using the same
conversion as SmolVLA's input (contract.py). Actions are not stored: scripts/convert_demos.py derives
them from consecutive TCP poses, so any policy's episodes (CheatCode, SmolVLA, teleop) share a format.
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from aic_smolvla import contract

INSERTION_TOPIC = "/scoring/insertion_event"


class EpisodeRecorder:
    def __init__(self, policy, get_observation, root, task, policy_name, hz=10.0):
        self.policy = policy
        self.get_observation = get_observation
        self.hz = hz
        self.task = task
        self.policy_name = policy_name
        port = (task.port_name or "port").replace("/", "_")
        self.dir = Path(root).expanduser() / f"{time.strftime('%Y%m%d_%H%M%S')}_{policy_name}_{port}"
        for cam in contract.CAMERAS:
            (self.dir / cam).mkdir(parents=True, exist_ok=True)
        self.states, self.sim_t, self.tcp_vel = [], [], []
        self.insertion_events = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def on_insertion_event(self, msg):
        self.insertion_events.append(msg.data)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        last_stamp = None
        while not self._stop.is_set():
            obs = self.get_observation()
            if obs is not None:
                s = obs.joint_states.header.stamp
                stamp = s.sec + s.nanosec * 1e-9
                if stamp != last_stamp:
                    last_stamp = stamp
                    self._save(obs, stamp)
            self.policy.sleep_for(1.0 / self.hz)

    def _save(self, obs, stamp):
        i = len(self.states)
        for cam in contract.CAMERAS:
            img = contract.image_to_numpy(getattr(obs, cam.replace("_camera", "_image")))
            cv2.imwrite(str(self.dir / cam / f"{i:06d}.jpg"), img[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.states.append(contract.state_to_numpy(obs))
        v = obs.controller_state.tcp_velocity
        self.tcp_vel.append([v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z])
        self.sim_t.append(stamp)

    def stop(self, result=None):
        self._stop.set()
        self._thread.join(timeout=5)
        n = len(self.states)
        np.savez_compressed(
            self.dir / "data.npz",
            state=np.asarray(self.states, dtype=np.float32).reshape(n, contract.STATE_DIM),
            sim_t=np.asarray(self.sim_t, dtype=np.float64),
            tcp_velocity=np.asarray(self.tcp_vel, dtype=np.float32).reshape(n, 6),
        )
        t = self.task
        meta = {
            "policy": self.policy_name,
            "prompt": contract.task_prompt(t),
            "task": {k: getattr(t, k) for k in ("id", "cable_type", "cable_name", "plug_type",
                                                 "plug_name", "port_type", "port_name",
                                                 "target_module_name", "time_limit")},
            "frames": n,
            "fps": self.hz,
            "sim_seconds": round(self.sim_t[-1] - self.sim_t[0], 2) if n > 1 else 0.0,
            "insertion_events": self.insertion_events,
            "inserted": any(self.insertion_events),
            "result": result,
            "state_names": contract.STATE_NAMES,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        self._write_preview(n)
        return self.dir, meta

    def _write_preview(self, n):
        if n == 0:
            return
        first = [cv2.imread(str(self.dir / c / "000000.jpg")) for c in contract.CAMERAS]
        h, w = first[0].shape[:2]
        out = cv2.VideoWriter(str(self.dir / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                              self.hz, (w * 3, h + 24))
        for i in range(n):
            row = np.hstack([cv2.imread(str(self.dir / c / f"{i:06d}.jpg")) for c in contract.CAMERAS])
            bar = np.zeros((24, row.shape[1], 3), np.uint8)
            label = f"{self.policy_name} | {contract.task_prompt(self.task)} | t={self.sim_t[i] - self.sim_t[0]:5.1f}s"
            cv2.putText(bar, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            out.write(np.vstack([bar, row]))
        out.release()


def subscribe_insertion(policy, recorder_ref):
    """Subscribe once to the scoring insertion event; forwards to the active recorder, if any."""
    from std_msgs.msg import String

    def cb(msg):
        rec = recorder_ref()
        if rec is not None:
            rec.on_insertion_event(msg)

    return policy._parent_node.create_subscription(String, INSERTION_TOPIC, cb, 10)
