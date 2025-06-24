import torch
from humanoidverse.envs.base_task.term import base
from humanoidverse.utils.torch_utils import quat_rotate_inverse
from humanoidverse.utils.torch_utils import quat_rotate_inverse, quat_apply
from isaac_utils.rotations import wrap_to_pi

class FeetRewards(base.BaseManager):
    def __init__(self, _task):
        super(FeetRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    #def _reward_penalty_slippage(self):
    def _reward_penalty_feet_slippage(self):
        # assert self.task.simulator._rigid_body_vel.shape[1] == 20
        robotdata_manager = self.task.robotdata_manager
        #feet_manager = self.task.feet_manager

        foot_vel = self.task.simulator._rigid_body_vel[:, robotdata_manager.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * \
                         (torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, :], dim=-1) > 1.), dim=1)


    ## contact_forces
    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        robotdata_manager = self.task.robotdata_manager
        return torch.any(torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, 2]), dim=1)

    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        robotdata_manager = self.task.robotdata_manager
        _forces = torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, :], dim=-1)
        _reward = (_forces - self.config.rewards.locomotion_max_contact_force).clip(min=0.)
        return torch.sum(_reward, dim=1)

    ## air
    def _reward_penalty_in_the_air(self):
        if not hasattr(self.task, "feet_manager"):
            return None

        feet_manager = self.task.feet_manager

        first_foot_contact = feet_manager.contact_filt[:, 0]
        second_foot_contact = feet_manager.contact_filt[:, 1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        if not hasattr(self.task, "feet_manager"):
            return None
        if not hasattr(self.task, "command_manager"):
            return None
        feet_manager = self.task.feet_manager
        command_manager = self.task.command_manager

        first_contact = (feet_manager.feet_air_time > self.task.dt) * feet_manager.contact_filt

        rew_airTime = torch.sum((feet_manager.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(command_manager.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        return rew_airTime

    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # Jiawei: Key ingredient.
        if not hasattr(self.task, "feet_manager"):
            return None
        feet_manager = self.task.feet_manager

        from_air_to_contact = torch.logical_and(feet_manager.contact_filt, ~feet_manager.last_contacts_filt)
        rew_feet_max_height = torch.sum(
            torch.clamp_min(self.config.rewards.desired_feet_max_height_for_this_air - feet_manager.feet_air_max_height, 0) * \
            from_air_to_contact, dim=1) # reward only on first contact with the ground

        return rew_feet_max_height

    ## close
    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        robotdata_manager = self.task.robotdata_manager

        left_foot_xy = self.task.simulator._rigid_body_pos[:, robotdata_manager.feet_indices[0], :2]
        right_foot_xy = self.task.simulator._rigid_body_pos[:, robotdata_manager.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0

    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        robotdata_manager = self.task.robotdata_manager

        left_knee_xy = self.task.simulator._rigid_body_pos[:, robotdata_manager.knee_indices[0], :2]
        right_knee_xy = self.task.simulator._rigid_body_pos[:, robotdata_manager.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0

    ## feet ori
    def _reward_penalty_feet_ori(self):
        robotdata_manager = self.task.robotdata_manager
        robotstatus_manager = self.task.robotstatus_manager

        left_quat = self.task.simulator._rigid_body_rot[:, robotdata_manager.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, robotstatus_manager.gravity_vec)
        right_quat = self.task.simulator._rigid_body_rot[:, robotdata_manager.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, robotstatus_manager.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5

    #def feet heading alignment(self):
    def _reward_feet_heading_alignment(self):
        robotstatus_manager = self.task.robotstatus_manager
        robotdata_manager = self.task.robotdata_manager

        left_quat = self.task.simulator._rigid_body_rot[:, robotdata_manager.feet_indices[0]]
        right_quat = self.task.simulator._rigid_body_rot[:, robotdata_manager.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, robotdata_manager.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, robotdata_manager.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(robotstatus_manager.base_quat, robotdata_manager.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))

        return heading_diff_left + heading_diff_right

    ## feet height
    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        robotdata_manager = self.task.robotdata_manager

        feet_height = self.task.simulator._rigid_body_pos[:, robotdata_manager.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable

    ## feet swing height
    def _reward_penalty_feet_swing_height(self):
        robotdata_manager = self.task.robotdata_manager

        contact = torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, :3], dim=2) > 1.
        feet_height = self.task.simulator._rigid_body_pos[:, robotdata_manager.feet_indices, 2]
        height_error = torch.square(feet_height - self.config.rewards.feet_height_target) * ~contact
        return torch.sum(height_error, dim=(1))

    ## feet swing height
    def _reward_walk_symmetry(self):
        robotdata_manager = self.task.robotdata_manager
        _diff = self.task.simulator.dof_pos - robotdata_manager.default_dof_pos

        _left = _diff[:, [0, 3]]
        _right = _diff[:, [6, 9]]

        _total = torch.mean(torch.abs(_left + _right))
        _left = torch.abs(torch.sum(_left, dim = -1))
        _right = torch.abs(torch.sum(_right, dim = -1))

        _mean = (_left + _right + _total)
        return torch.exp(-_mean)
