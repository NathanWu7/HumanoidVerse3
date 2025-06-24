from humanoidverse.envs.base_task.term import base
import numpy as np
import torch

class NoiseCurrculum(base.BaseManager):

    def __init__(self, _task):
        super(NoiseCurrculum, self).__init__(_task)

        if 'add_noise_currculum' in self.config.obs:
            self.add_noise_currculum = self.config.obs.add_noise_currculum
            self.current_noise_curriculum_value = self.config.obs.noise_initial_value
        else:
            self.add_noise_currculum = False

    def reset(self, env_ids):
        if 0 == len(env_ids):
            return

        if not self.add_noise_currculum:
            return

        # waitting for episode reset
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager

        if episode_manager.average_episode_length < self.config.obs.soft_dof_pos_curriculum_level_down_threshold:
            self.current_noise_curriculum_value *= (1 - self.config.obs.soft_dof_pos_curriculum_degree)
        elif episode_manager.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.current_noise_curriculum_value *= (1 + self.config.obs.soft_dof_pos_curriculum_degree)

        self.current_noise_curriculum_value = np.clip(self.current_noise_curriculum_value,
                                                      self.config.obs.noise_value_min,
                                                      self.config.obs.noise_value_max)

        if hasattr(self.task, "observations_manager"):
            observations_manager = self.task.observations_manager

            for _key in self.config.obs.noise_scales:
                observations_manager.noise_scales[_key] = self.current_noise_curriculum_value * self.config.obs.noise_scales[_key]

    def post_compute(self):
        if not self.add_noise_currculum:
            return
        if hasattr(self.task, "extras_manager"):
            self.task.extras_manager.log_dict["current_noise_curriculum_value"] = torch.tensor(self.current_noise_curriculum_value, dtype=torch.float)
