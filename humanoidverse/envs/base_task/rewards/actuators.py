import torch
from humanoidverse.envs.base_task.term import base

class ActuatorsRewards(base.BaseManager):
    def __init__(self, _task):
        super(ActuatorsRewards, self).__init__(_task)

    def init(self):
        _robotdata_manager = self.task.robotdata_manager

        # currculum
        self.limits_dof_pos = self.task.simulator.dof_pos_limits
        self.limits_dof_vel = _robotdata_manager.dof_vel_limits * self.config.rewards.reward_limit.soft_dof_vel_limit
        self.limits_torque = _robotdata_manager.torque_limits * self.config.rewards.reward_limit.soft_torque_limit

    ########################### PENALTY REWARDS ###########################
    def _reward_penalty_torques(self):
        # Penalize torques
        _actuators_manager = self.task.actuators_manager
        return torch.sum(torch.square(_actuators_manager.torques), dim=1)


    ######################## LIMITS REWARDS #########################
    def _reward_limits_torque(self):
        # penalize torques too close to the limit
        _actuators_manager = self.task.actuators_manager
        return torch.sum((torch.abs(_actuators_manager.torques) - self.limits_torque).clip(min=0.), dim=1)

    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.task.simulator.dof_pos - self.limits_dof_pos[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.task.simulator.dof_pos - self.limits_dof_pos[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.task.simulator.dof_vel) - self.limits_dof_vel).clip(min=0., max=1.), dim=1)
