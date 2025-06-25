import torch
from humanoidverse.envs.base_task.term.status import robotstatus
from humanoidverse.envs.motion_tracking.asap.status import asap_robotstatus_extend
from tools.isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
)

class AsapStatus(robotstatus.StatusManager):
    def __init__(self, _task):
        super(AsapStatus, self).__init__(_task)
        self.extends = asap_robotstatus_extend.AsapMotionExtends(_task)

    def init(self):
        super(AsapStatus, self).init()
        # for debug settings TODO check later for must load
        if self.task.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.task.simulator.add_visualize_entities(num_visualize_markers)
        elif self.task.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.task.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass

        self.extends.init()

    @property
    def num_bodies(self):
        robotdata_manager = self.task.robotdata_manager
        _num_bodies = robotdata_manager.num_bodies + self.extends.num_bodies
        return _num_bodies

    def pre_compute(self):
        super(AsapStatus, self).pre_compute()
        ## rigid_body
        self._rigid_body_pos = self.task.simulator._rigid_body_pos
        self._rigid_body_rot = self.task.simulator._rigid_body_rot
        self._rigid_body_ang_vel = self.task.simulator._rigid_body_ang_vel
        self._rigid_body_vel = self.task.simulator._rigid_body_vel
        self.extends.pre_compute(self)

    def post_compute(self):
        '''
        for log
        '''
        if not hasattr(self.task, "extras_manager"):
            return

        extras_manager = self.task.extras_manager
        robotdata_manager = self.task.robotdata_manager

        if not hasattr(robotdata_manager, "current_motion_ref"):
            return

        ref_body_pos = robotdata_manager.current_motion_ref["rg_pos_t"]
        ref_joint_pos = robotdata_manager.current_motion_ref["dof_pos"] # [num_envs, num_markers, 3]
        ## diff compute - kinematic position
        _dif_global_body_pos = ref_body_pos - self._rigid_body_pos

        if hasattr(robotdata_manager, "upper_body_id"):
            upper_body_diff = _dif_global_body_pos[:, robotdata_manager.upper_body_id, :]
            upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
            extras_manager.log_dict["upper_body_diff_norm"] = upper_body_diff_norm


        if hasattr(robotdata_manager, "lower_body_id"):
            lower_body_diff = _dif_global_body_pos[:, robotdata_manager.lower_body_id, :]
            lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
            extras_manager.log_dict["lower_body_diff_norm"] = lower_body_diff_norm

        if hasattr(robotdata_manager, "motion_tracking_id"):
            vr_3point_diff = _dif_global_body_pos[:, robotdata_manager.motion_tracking_id, :]
            vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
            extras_manager.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm

        joint_pos_diff = ref_joint_pos - self.task.simulator.dof_pos
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()
        extras_manager.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm

    ######################### Observations #########################
    ## rigid body local pos
    def _get_obs_dif_local_rigid_body_pos(self):
        robotdata_manager = self.task.robotdata_manager
        ref_body_pos = robotdata_manager.next_motion_ref["rg_pos_t"]
        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        #heading_rot = calc_heading_quat(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        #heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        dif_global_body_pos_for_obs_compute = ref_body_pos.view(self.num_envs, -1, 3) - \
                                              self._rigid_body_pos.view(self.num_envs, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), \
                                                 dif_global_body_pos_for_obs_compute.view(-1, 3))

        return dif_local_body_pos_flat.view(self.num_envs, -1) # (num_envs, num_rigid_bodies*3)

    def _get_obs_local_ref_rigid_body_pos(self):
        robotdata_manager = self.task.robotdata_manager
        ref_body_pos = robotdata_manager.next_motion_ref["rg_pos_t"]

        global_ref_rigid_body_pos = ref_body_pos.view(self.num_envs, -1, 3) - \
            self.task.simulator.robot_root_states[:, :3].view(self.num_envs, 1, 3)  # preserves the body position

        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        return local_ref_rigid_body_pos_flat.view(self.num_envs, -1) # (num_envs, num_rigid_bodies * 3)

    def _get_obs_vr_3point_pos(self):
        robotdata_manager = self.task.robotdata_manager
        assert hasattr(robotdata_manager, 'motion_tracking_id')
        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)

        ref_body_pos = robotdata_manager.next_motion_ref["rg_pos_t"]
        ref_vr_3point_pos = ref_body_pos.view(self.num_envs, -1, 3)[:, robotdata_manager.motion_tracking_id, :]

        vr_2root_pos = ref_vr_3point_pos - self.task.simulator.robot_root_states[:, 0:3].view((self.num_envs, 1, 3))
        return my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(self.num_envs, -1)

    # rigid body local vel
    def _get_obs_local_ref_rigid_body_vel(self):
        robotdata_manager = self.task.robotdata_manager
        ref_body_vel = robotdata_manager.next_motion_ref["body_vel_t"] # [num_envs, num_markers, 3]

        # self.base_quat ?
        heading_inv_rot = calc_heading_quat_inv(self.task.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_bodies, -1).reshape(-1, 4)

        global_ref_body_vel = ref_body_vel.view(self.num_envs, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))

        return local_ref_rigid_body_vel_flat.view(self.num_envs, -1) # (num_envs, num_rigid_bodies * 3)

    # phase
    #################### Deepmimic phase ######################
    def _get_obs_ref_motion_phase(self):
        robotdata_manager = self.task.robotdata_manager
        _ref_motion_length = robotdata_manager.motion_len
        _ref_motion_phase = robotdata_manager.next_motion_times / _ref_motion_length
        return _ref_motion_phase.unsqueeze(1)
