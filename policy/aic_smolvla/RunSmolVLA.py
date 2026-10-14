"""AIC policy that drives the robot with SmolVLA (served by aic_smolvla.server).

Loaded by AIC's aic_model node (AIC pixi env):
    PYTHONPATH=~/eee598-aic/policy pixi run ros2 run aic_model aic_model --ros-args \
        -p use_sim_time:=true -p policy:=aic_smolvla.RunSmolVLA

Receding-horizon loop: get the latest observation, ask the server for a 50-step action chunk,
execute the first `exec_steps` twists at `hz`, repeat until `max_seconds`. Timing uses the node's
clock (simulation time), like AIC's example policies, so behavior doesn't depend on how fast
Gazebo runs. Every twist is clipped
(speed limits below), and the trial stops early (arm halted) if the TCP drifts more than
MAX_DRIFT from where it started or the wrist force changes by more than MAX_FORCE. The arm is
stopped on exit.

Settings (environment variables):
    AIC_SMOLVLA_PORT         server port (50765)
    AIC_SMOLVLA_MAX_LIN      max linear speed, m/s (0.02)
    AIC_SMOLVLA_MAX_ANG      max angular speed, rad/s (0.1)
    AIC_SMOLVLA_HZ           command rate (10)
    AIC_SMOLVLA_EXEC_STEPS   actions executed per chunk before re-planning (10)
    AIC_SMOLVLA_MAX_SECONDS  sim-time seconds per trial, capped by the task's time limit (30)
    AIC_SMOLVLA_MAX_DRIFT    max TCP distance from its start position, m (0.05)
    AIC_SMOLVLA_MAX_FORCE    max change in wrist force from its start value, N (15)
    AIC_SMOLVLA_FRAME        twist frame: base_link or gripper/tcp (base_link)
    AIC_SMOLVLA_LOG_DIR      per-trial JSONL logs (~/eee598-aic/outputs/smolvla_runs)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench

from aic_smolvla import contract, transport


def _env(name, default, cast=float):
    return cast(os.environ.get(f"AIC_SMOLVLA_{name}", default))


class RunSmolVLA(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.max_lin = _env("MAX_LIN", 0.02)
        self.max_ang = _env("MAX_ANG", 0.1)
        self.hz = _env("HZ", 10.0)
        self.exec_steps = _env("EXEC_STEPS", 10, int)
        self.max_seconds = _env("MAX_SECONDS", 30.0)
        self.max_drift = _env("MAX_DRIFT", 0.05)
        self.max_force = _env("MAX_FORCE", 15.0)
        self.frame = _env("FRAME", "base_link", str)
        self.log_dir = Path(
            os.path.expanduser(_env("LOG_DIR", "~/eee598-aic/outputs/smolvla_runs", str))
        )

        address = (transport.DEFAULT_ADDRESS[0], _env("PORT", transport.DEFAULT_ADDRESS[1], int))
        deadline = time.monotonic() + 60
        while True:
            try:
                self.conn = transport.connect(address)
                break
            except (ConnectionRefusedError, OSError):
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"No SmolVLA server on {address}; start scripts/run_smolvla_server.sh first"
                    )
                time.sleep(1.0)
        info = transport.request(self.conn, {"cmd": "ping"})["info"]
        self.chunk_size = info["chunk_size"]
        self.get_logger().info(
            f"RunSmolVLA connected to server {address}: {info} | limits {self.max_lin} m/s, "
            f"{self.max_ang} rad/s | {self.hz} Hz, {self.exec_steps} steps/chunk, frame {self.frame}"
        )

    def _twist_update(self, action) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = Twist(
            linear=Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2])),
            angular=Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5])),
        )
        msg.target_stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten()
        msg.target_damping = np.diag([40.0, 40.0, 40.0, 15.0, 15.0, 15.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    @staticmethod
    def _tcp_and_force(obs_msg):
        p = obs_msg.controller_state.tcp_pose.position
        f = obs_msg.wrist_wrench.wrench.force
        return np.array([p.x, p.y, p.z]), np.array([f.x, f.y, f.z])

    def _guard(self, obs_msg, start):
        """Return a reason string if the safety limits are exceeded, else None."""
        tcp, force = self._tcp_and_force(obs_msg)
        drift = float(np.linalg.norm(tcp - start[0]))
        dforce = float(np.linalg.norm(force - start[1]))
        if drift > self.max_drift:
            return f"TCP drifted {drift * 1000:.0f} mm (limit {self.max_drift * 1000:.0f} mm)"
        if dforce > self.max_force:
            return f"wrist force changed {dforce:.1f} N (limit {self.max_force:.0f} N)"
        return None

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        prompt = contract.task_prompt(task)
        run_for = self.max_seconds
        if task.time_limit:
            run_for = min(run_for, max(1.0, float(task.time_limit) - 5.0))
        self.get_logger().info(f"RunSmolVLA.insert_cable: '{prompt}' for {run_for:.0f}s")
        transport.request(self.conn, {"cmd": "reset"})

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{task.id or 'task'}.jsonl"
        period = 1.0 / self.hz
        t_start = self.time_now()
        elapsed = lambda: (self.time_now() - t_start).nanoseconds * 1e-9  # sim seconds
        wall_start = time.monotonic()
        chunks, sent, latencies = 0, 0, []
        start_state, stop_reason = None, None

        with open(log_path, "w") as log:
            log.write(json.dumps({"event": "start", "task": prompt, "time_limit": task.time_limit}) + "\n")
            try:
                while elapsed() < run_for and not stop_reason:
                    obs_msg = get_observation()
                    if obs_msg is None:
                        self.sleep_for(0.05)
                        continue
                    if start_state is None:
                        start_state = self._tcp_and_force(obs_msg)
                    obs = contract.observation_to_numpy(obs_msg)
                    reply = transport.request(self.conn, {"cmd": "act", "obs": obs, "task": prompt})
                    chunk = reply["actions"]
                    chunks += 1
                    latencies.append(reply["latency_s"])
                    log.write(json.dumps({
                        "event": "chunk",
                        "t": round(elapsed(), 3),
                        "wall_t": round(time.monotonic() - wall_start, 3),
                        "sim_t": round(obs_msg.joint_states.header.stamp.sec
                                       + obs_msg.joint_states.header.stamp.nanosec * 1e-9, 3),
                        "latency_s": round(reply["latency_s"], 4),
                        "tcp": obs["state"][:7].round(4).tolist(),
                        "raw_first": np.round(chunk[0], 4).tolist(),
                    }) + "\n")

                    for action in chunk[: self.exec_steps]:
                        tick = elapsed()
                        latest = get_observation()
                        stop_reason = latest is not None and self._guard(latest, start_state)
                        if stop_reason:
                            move_robot(motion_update=self._twist_update(np.zeros(contract.ACTION_DIM)))
                            self.get_logger().warn(f"RunSmolVLA safety stop: {stop_reason}")
                            log.write(json.dumps({"event": "safety_stop", "reason": stop_reason,
                                                  "t": round(elapsed(), 3)}) + "\n")
                            break
                        move_robot(motion_update=self._twist_update(
                            contract.clip_twist(action, self.max_lin, self.max_ang)))
                        sent += 1
                        if elapsed() >= run_for:
                            break
                        remaining = period - (elapsed() - tick)
                        if remaining > 0:
                            self.sleep_for(remaining)
                    send_feedback(f"smolvla chunk {chunks}, {sent} commands")
            finally:
                move_robot(motion_update=self._twist_update(np.zeros(contract.ACTION_DIM)))
                summary = {
                    "event": "end",
                    "sim_seconds": round(elapsed(), 2),
                    "wall_seconds": round(time.monotonic() - wall_start, 2),
                    "chunks": chunks,
                    "commands": sent,
                    "safety_stop": stop_reason or None,
                    "latency_median_s": float(np.median(latencies)) if latencies else None,
                    "latency_max_s": float(np.max(latencies)) if latencies else None,
                }
                log.write(json.dumps(summary) + "\n")
                self.get_logger().info(f"RunSmolVLA done: {summary} | log {log_path}")
        return True
