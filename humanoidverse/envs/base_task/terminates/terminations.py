import torch
from humanoidverse.envs.base_task.term import base

class TerminateManager(base.BaseManager):
    def __init__(self, _task):
        super().__init__(_task)
        self.termination_gravity_x = self.config.termination_scales.termination_gravity_x
        self.termination_gravity_y = self.config.termination_scales.termination_gravity_y
        if hasattr(self.task, 'task_data'):
            self.task_data = self.task.task_data

    def init(self):
        pass

    def _check_terminate_by_contact(self):
        robotdata_manager = self.task.robotdata_manager
        task = self.task
        return torch.any(torch.norm(task.simulator.contact_forces[:, robotdata_manager.termination_contact_indices, :], dim=-1) > 1., dim=1)

    def _check_terminate_by_gravity(self):
        robotstatus_manager = self.task.robotstatus_manager
        _buf = torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 0:1]) > self.termination_gravity_x, dim=1)
        _buf |= torch.any(torch.abs(robotstatus_manager.projected_gravity[:, 1:2]) > self.termination_gravity_y, dim=1)
        return _buf

    def _check_terminate_by_low_height(self):
        task = self.task
        return torch.any(task.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

    def _check_terminate_when_close_to_dof_pos_limit(self):
        if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
            task = self.task
            out_of_dof_pos_limits = -(task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (task.simulator.dof_pos - task.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)

            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            return out_of_dof_pos_limits > 0.

        return None

    def _check_terminate_when_close_to_dof_vel_limit(self):
        if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
            task = self.task
            out_of_dof_vel_limits = torch.sum((torch.abs(task.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)

            return  out_of_dof_vel_limits > 0.

        return None

    def _check_terminate_when_close_to_torque_limit(self):
        if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)
            return  out_of_torque_limits > 0.
        return None

    def post_compute(self):
        if not self.config.termination_curriculum:
            return

        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager

        self.gravity_terminate_buf = torch.zeros_like(self.task.episode_manager.reset_buf)

        if self.config.termination_curriculum.get("terminate_by_gravity_curriculum", False):
            if episode_manager.average_episode_length < self.config.termination_curriculum.termination_gravity_curriculum_level_down_threshold:
                self.termination_gravity_x *= (1 + self.config.termination_curriculum.termination_gravity_curriculum_degree)
                self.termination_gravity_y *= (1 + self.config.termination_curriculum.termination_gravity_curriculum_degree)
            elif episode_manager.average_episode_length > self.config.termination_curriculum.termination_gravity_curriculum_level_up_threshold:
                self.termination_gravity_x *= (1 + self.config.termination_curriculum.termination_gravity_curriculum_degree)
                self.termination_gravity_y *= (1 + self.config.termination_curriculum.termination_gravity_curriculum_degree)

            self.termination_gravity_x = max(self.termination_gravity_x, self.config.termination_curriculum.termination_gravity_x_min_limit)
            self.termination_gravity_x = min(self.termination_gravity_x, self.config.termination_curriculum.termination_gravity_x_max_limit)
            self.termination_gravity_y = max(self.termination_gravity_y, self.config.termination_curriculum.termination_gravity_y_min_limit)
            self.termination_gravity_y = min(self.termination_gravity_y, self.config.termination_curriculum.termination_gravity_y_max_limit)
            ## update

            assert hasattr(self.task, "extras_manager")
            extras_manager = self.task.extras_manager

            extras_manager.log_dict["termination_gravity_x"] = torch.tensor(self.termination_gravity_x, device = self.task.device)
            extras_manager.log_dict["termination_gravity_y"] = torch.tensor(self.termination_gravity_y, device = self.task.device)
