from humanoidverse.envs.base_task.term import base
from humanoidverse.envs.base_task.actions import lpf

import numpy as np
import torch

'''
config file:

robot:
  control:
    action_filt:
    action_cutfreq:

'''

class FilterManager(base.BaseManager):
    def __init__(self, _task):
        super(FilterManager, self).__init__(_task)

    # stage 1
    def init(self):
        if not self.config.robot.control.action_filt:
            return

        _dim_actions = self.task.config.robot.actions_dim

        self.action_filter = lpf.ActionFilterButterTorch(
            lowcut=np.zeros(self.num_envs * _dim_actions),
            highcut=np.ones(self.num_envs * _dim_actions) * self.config.robot.control.action_cutfreq,
            sampling_rate=1./self.dt, num_joints=self.num_envs * _dim_actions,
            device=self.device)

    def pre_physics_step(self, actions):
        if not self.config.robot.control.action_filt:
            return actions

        return self.action_filter.filter(actions.reshape(-1)).reshape(self.num_envs, -1)

    # stage 3
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        _dim_actions = self.task.config.robot.actions_dim

        filter_action_ids_torch = torch.concat([torch.arange(_dim_actions, dtype=torch.int32, device=self.device) + env_id * _dim_actions for env_id in env_ids])
        self.action_filter.reset_hist(filter_action_ids_torch)
