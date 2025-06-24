import torch
from humanoidverse.envs.base_task.term import base

class StatusRewards(base.BaseManager):
    def __init__(self, _task):
        super(StatusRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    # dof
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        #_robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square(self.task.simulator.dof_vel), dim=1)

    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square((_robotstatus_manager.last_dof_vel - self.task.simulator.dof_vel) / self.task.dt), dim=1)

    # base
    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.square(_robotstatus_manager.base_lin_vel[:, 2])

    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square(_robotstatus_manager.base_ang_vel[:, :2]), dim=1)

    # ori
    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        _robotstatus_manager = self.task.robotstatus_manager
        return torch.sum(torch.square(_robotstatus_manager.projected_gravity[:, :2]), dim=1)
