import torch
from humanoidverse.envs.base_task.term import base

class ASAPStatusRewards(base.BaseManager):
    def __init__(self, _task):
        super(ASAPStatusRewards, self).__init__(_task)

    def _reward_teleop_body_position_extend(self):
        _robotstatus_manager = self.task.robotstatus_manager
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_body_pos = _robotdata_manager.current_motion_ref["rg_pos_t"]
        dif_global_body_pos = ref_body_pos - _robotstatus_manager._rigid_body_pos

        upper_body_diff = dif_global_body_pos[:, _robotdata_manager.upper_body_id, :]
        lower_body_diff = dif_global_body_pos[:, _robotdata_manager.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight

        return r_body_pos

    def _reward_teleop_vr_3point(self):
        _robotstatus_manager = self.task.robotstatus_manager
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_body_pos = _robotdata_manager.current_motion_ref["rg_pos_t"]
        dif_global_body_pos = ref_body_pos - _robotstatus_manager._rigid_body_pos

        vr_3point_diff = dif_global_body_pos[:, _robotdata_manager.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
        return r_vr_3point

    def _reward_teleop_body_position_feet(self):
        _robotstatus_manager = self.task.robotstatus_manager
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_body_pos = _robotdata_manager.current_motion_ref["rg_pos_t"]
        dif_global_body_pos = ref_body_pos - _robotstatus_manager._rigid_body_pos

        feet_diff = dif_global_body_pos[:, _robotdata_manager.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)
        return r_feet

    def _reward_teleop_body_rotation_extend(self):
        _robotstatus_manager = self.task.robotstatus_manager
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_body_rot = _robotdata_manager.current_motion_ref["rg_rot_t"] # [num_envs, num_markers, 4]
        rotation_diff = ref_body_rot - _robotstatus_manager._rigid_body_rot

        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        return r_body_rot

    def _reward_teleop_body_velocity_extend(self):
        _robotstatus_manager = self.task.robotstatus_manager
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_body_vel = _robotdata_manager.current_motion_ref["body_vel_t"] # [num_envs, num_markers, 3]
        velocity_diff = ref_body_vel - _robotstatus_manager._rigid_body_vel

        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        return r_body_vel

    def _reward_teleop_body_ang_velocity_extend(self):
        _robotstatus_manager = self.task.robotstatus_manager
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_body_ang_vel = _robotdata_manager.current_motion_ref["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ang_velocity_diff = ref_body_ang_vel - _robotstatus_manager._rigid_body_ang_vel

        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        return r_body_ang_vel

    def _reward_teleop_joint_position(self):
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_joint_pos = _robotdata_manager.current_motion_ref["dof_pos"] # [num_envs, num_dofs]

        joint_pos_diff = ref_joint_pos - self.task.simulator.dof_pos
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)
        return r_joint_pos

    def _reward_teleop_joint_velocity(self):
        _robotdata_manager = self.task.robotdata_manager
        if not hasattr(_robotdata_manager, 'current_motion_ref'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        ref_joint_vel = _robotdata_manager.current_motion_ref["dof_vel"] # [num_envs, num_dofs]
        joint_vel_diff = ref_joint_vel - self.task.simulator.dof_vel

        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        return r_joint_vel
