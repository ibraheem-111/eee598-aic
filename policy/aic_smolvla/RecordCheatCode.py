"""AIC's CheatCode policy (ground-truth poses), recorded as demonstrations for fine-tuning SmolVLA.

Needs the simulator started with ground_truth:=true. Run with scripts/record_demos.sh.
Episodes go to $AIC_RECORD_DIR (default ~/eee598-aic/data/demos/raw); see recording.py.
"""

import os

from aic_example_policies.ros.CheatCode import CheatCode

from aic_smolvla.recording import EpisodeRecorder, subscribe_insertion


class RecordCheatCode(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.record_root = os.environ.get("AIC_RECORD_DIR", "~/eee598-aic/data/demos/raw")
        self._recorder = None
        self._insertion_sub = subscribe_insertion(self, lambda: self._recorder)

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self._recorder = EpisodeRecorder(self, get_observation, self.record_root, task,
                                         "cheatcode").start()
        result = None
        try:
            result = super().insert_cable(task, get_observation, move_robot, send_feedback)
            return result
        finally:
            folder, meta = self._recorder.stop(result)
            self._recorder = None
            self.get_logger().info(
                f"Recorded {meta['frames']} frames ({meta['sim_seconds']}s), "
                f"inserted={meta['inserted']} -> {folder}"
            )
