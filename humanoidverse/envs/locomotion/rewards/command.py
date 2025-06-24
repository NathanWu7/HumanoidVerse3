import torch
from humanoidverse.envs.base_task.term import base

class CommandRewards(base.BaseManager):
    def __init__(self, _task):
        super(CommandRewards, self).__init__(_task)

    ########################### TRACKING REWARDS ###########################
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        if not hasattr(self.task, "command_manager"):
            return None

        command_manager = self.task.command_manager
        robotstatus_manager = self.task.robotstatus_manager

        lin_vel_error = torch.sum(torch.square(command_manager.commands[:, :2] - robotstatus_manager.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.config.rewards.reward_tracking_sigma.lin_vel)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        if not hasattr(self.task, "command_manager"):
            return None

        command_manager = self.task.command_manager
        robotstatus_manager = self.task.robotstatus_manager
        ang_vel_error = torch.square(command_manager.commands[:, 2] - robotstatus_manager.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.config.rewards.reward_tracking_sigma.ang_vel)
