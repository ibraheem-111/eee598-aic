"""AIC policy for residual RL: the base policy (SmolVLA/ACT on the server) plus a SAC residual (rl_agent).

Run with the simulator's ground truth on (`ground_truth:=true`) and the server started with --rl-dir
(scripts/rl_train.sh does both). One RL step = one action chunk. Ground-truth TF is used only for the
reward and for the critic's privileged input; the actor sees robot state, force and the base chunk.

Reward per step (shaped by potential: progress toward the port, so the optimum is unchanged):
    100 * (d_prev - d)        plug-to-port distance decrease, in cm
    - FORCE_PENALTY           if the wrist force changed by more than FORCE_LIMIT N (AIC penalizes > 20 N)
    + INSERT_BONUS, terminal  on a scoring insertion event
    - STOP_PENALTY, terminal  on a safety stop

Settings: the AIC_SMOLVLA_* limits of RunSmolVLA, plus
    AIC_RL_EVAL=1             deterministic actor, nothing stored or trained
    AIC_RL_ZERO_RESIDUAL=1    base policy only (implies AIC_RL_EVAL): logs true distances, e.g. coverage tests
    AIC_RL_FORCE_LIMIT (20), AIC_RL_FORCE_PENALTY (0.5), AIC_RL_INSERT_BONUS (50), AIC_RL_STOP_PENALTY (10)
"""

import json
import os
import time

import numpy as np
from rclpy.time import Time
from std_msgs.msg import String

from aic_smolvla import contract, transport
from aic_smolvla.recording import INSERTION_TOPIC, EpisodeRecorder
from aic_smolvla.RunSmolVLA import RunSmolVLA


def _rl_env(name, default):
    return float(os.environ.get(f"AIC_RL_{name}", default))


class RunResidualRL(RunSmolVLA):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.zero_residual = os.environ.get("AIC_RL_ZERO_RESIDUAL") == "1"
        self.eval_mode = os.environ.get("AIC_RL_EVAL") == "1" or self.zero_residual
        self.force_limit = _rl_env("FORCE_LIMIT", 20)
        self.force_penalty = _rl_env("FORCE_PENALTY", 0.5)
        self.insert_bonus = _rl_env("INSERT_BONUS", 50)
        self.stop_penalty = _rl_env("STOP_PENALTY", 10)
        self._inserted = False
        self._rl_sub = parent_node.create_subscription(String, INSERTION_TOPIC, self._on_insertion, 10)

    def _on_insertion(self, msg):  # the recorder gets the event through RunSmolVLA's own subscription
        self._inserted = True

    def _xyz(self, frame):
        t = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time()).transform.translation
        return np.array([t.x, t.y, t.z])

    def _priv(self, task, tcp):
        plug = self._xyz(f"{task.cable_name}/{task.plug_name}_link")
        port = self._xyz(f"task_board/{task.target_module_name}/{task.port_name}_link")
        d = float(np.linalg.norm(plug - port))
        return np.concatenate([plug - tcp, port - tcp, [d]]).astype(np.float32), d

    def _wait_priv(self, task, tcp, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self._priv(task, tcp)
            except Exception as ex:  # TF not there yet
                if time.monotonic() > deadline:
                    raise RuntimeError(f"no ground-truth TF (run the sim with ground_truth:=true): {ex}")
                self.sleep_for(0.2)

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs) -> bool:
        prompt = contract.task_prompt(task)
        fields = contract.task_fields(task)
        run_for = self.max_seconds
        if task.time_limit:
            run_for = min(run_for, max(1.0, float(task.time_limit) - 5.0))
        transport.request(self.conn, {"cmd": "reset"})
        self._inserted = False
        if self.record_root:
            self._recorder = EpisodeRecorder(self, get_observation, self.record_root, task, "smolvla_rl").start()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_rl_{task.id or 'task'}.jsonl"
        period = 1.0 / self.hz
        t_start = self.time_now()
        elapsed = lambda: (self.time_now() - t_start).nanoseconds * 1e-9

        obs_msg = None
        while obs_msg is None:
            obs_msg = get_observation()
            if obs_msg is None:
                self.sleep_for(0.05)
        start_state = self._tcp_and_force(obs_msg)
        priv, d = self._wait_priv(task, start_state[0])
        d0, d_min, ret, reward, steps, stop_reason, force_steps = d, d, 0.0, 0.0, 0, None, 0
        log = open(log_path, "w")
        log.write(json.dumps({"event": "start", "task": prompt, "d0": d0, "eval": self.eval_mode}) + "\n")
        try:
            while elapsed() < run_for and not stop_reason and not self._inserted:
                obs = contract.observation_to_numpy(obs_msg)
                reply = transport.request(self.conn, {
                    "cmd": "rl_act", "obs": obs, "task": prompt, "task_fields": fields, "priv": priv,
                    "reward": reward, "t_frac": min(1.0, elapsed() / run_for),
                    "exec_steps": self.exec_steps, "deterministic": self.eval_mode,
                    "zero_residual": self.zero_residual})
                if "error" in reply:
                    raise RuntimeError(f"server error: {reply['error']}")
                for action in reply["actions"][: self.exec_steps]:
                    tick = elapsed()
                    latest = get_observation()
                    stop_reason = latest is not None and self._guard(latest, start_state)
                    if stop_reason or self._inserted:
                        break
                    move_robot(motion_update=self._twist_update(
                        contract.clip_twist(action, self.max_lin, self.max_ang)))
                    if elapsed() >= run_for:
                        break
                    remaining = period - (elapsed() - tick)
                    if remaining > 0:
                        self.sleep_for(remaining)
                steps += 1
                obs_msg = get_observation() or obs_msg
                tcp, force = self._tcp_and_force(obs_msg)
                priv, d_new = self._priv(task, tcp)
                over = float(np.linalg.norm(force - start_state[1])) > self.force_limit
                force_steps += over
                reward = 100.0 * (d - d_new) - (self.force_penalty if over else 0.0)
                if self._inserted:
                    reward += self.insert_bonus
                if stop_reason:
                    reward -= self.stop_penalty
                d, d_min, ret = d_new, min(d_min, d_new), ret + reward
                log.write(json.dumps({"event": "step", "t": round(elapsed(), 2), "d": round(d, 4),
                                      "reward": round(reward, 3), "residual": np.round(reply["residual"], 4).tolist()}) + "\n")
                send_feedback(f"rl step {steps} d={d * 100:.1f} cm")
        finally:
            move_robot(motion_update=self._twist_update(np.zeros(contract.ACTION_DIM)))
            summary = {"task": prompt, "port_type": fields.get("port_type"), "steps": steps,
                       "return": round(ret, 3), "d0": round(d0, 4), "d_final": round(d, 4),
                       "d_min": round(d_min, 4), "inserted": self._inserted, "force_steps": force_steps,
                       "safety_stop": stop_reason or None, "sim_seconds": round(elapsed(), 1),
                       "zero_residual": self.zero_residual}
            reply = transport.request(self.conn, {
                "cmd": "rl_end", "reward": reward, "priv": priv, "summary": summary,
                "terminal": bool(self._inserted or stop_reason), "deterministic": self.eval_mode})
            log.write(json.dumps({"event": "end", **summary, "agent": reply.get("entry")}) + "\n")
            log.close()
            self.get_logger().info(f"RunResidualRL done: {summary}")
            if self._recorder:
                self._recorder.stop(summary)
                self._recorder = None
        return True
