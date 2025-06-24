from humanoidverse.envs.base_task.term import base
from loguru import logger
import numpy as np
import torch

class MotionfarCurrculum(base.BaseManager):

    def __init__(self, _task):
        super(MotionfarCurrculum, self).__init__(_task)

    # stage 1
    def init(self):
        if not self.config.termination.terminate_when_motion_far or \
           not self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            return

        self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
        logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

    def pre_compute(self):
        if not self.config.termination.terminate_when_motion_far or \
           not self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            return

        assert hasattr(self.task, 'episode_manager')
        episode_manager = self.task.episode_manager

        episode_manager.terminate_when_motion_far_threshold = self.terminate_when_motion_far_threshold

    def reset(self, env_ids):
        if 0 == len(env_ids):
            return

        if not self.config.termination.terminate_when_motion_far or \
           not self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            return

        if not hasattr(self.task, 'episode_manager'):
            return
        episode_manager = self.task.episode_manager

        if not hasattr(self.task, 'asap_termination'):
            return

        if episode_manager.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif episode_manager.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)

        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold,
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min,
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)

        self.task.asap_termination.terminate_when_motion_far_threshold = self.terminate_when_motion_far_threshold

    def post_compute(self):
        if not self.config.termination.terminate_when_motion_far or \
           not self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            return

        if not hasattr(self.task, "extras_manager"):
            return

        extras_manager = self.task.extras_manager
        extras_manager.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)

