import torch
from humanoidverse.envs.base_task.term import base

class ActionsRewards(base.BaseManager):
    def __init__(self, _task):
        super(ActionsRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions
        _actions_manager = self.task.actions_manager
        return torch.sum(torch.square(_actions_manager.last_actions - _actions_manager.obs_actions), dim=1)
