from humanoidverse.envs.base_task.term import base
import torch
from tools.isaac_utils.rotations import (
    my_quat_rotate,
    quat_mul,
)

class AsapMotionExtends(base.BaseManager):
    def __init__(self, _task):
        super(AsapMotionExtends, self).__init__(_task)

    def init(self):
        if "extend_config" not in self.config.robot.motion:
            return

        extend_parent_ids = []
        extend_pos = []
        extend_rot = []

        for extend_config in self.config.robot.motion.extend_config:
            extend_parent_ids.append(self.task.simulator._body_list.index(extend_config["parent_name"]))
            # extend_parent_ids.append(self.task.simulator.find_rigid_body_indice(extend_config["parent_name"]))
            extend_pos.append(extend_config["pos"])
            extend_rot.append(extend_config["rot"])

        self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
        self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
        _extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
        self.extend_body_rot_in_parent_xyzw = _extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
        self.num_extend_bodies = len(extend_parent_ids)

    def pre_compute(self, _status):
        if "extend_config" not in self.config.robot.motion:
            return

        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate(
            self.task.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + _status._rigid_body_pos[:, self.extend_body_parent_ids]

        _status._rigid_body_pos = torch.cat([_status._rigid_body_pos, extend_curr_pos], dim=1)
        ################### EXTEND Rigid body Rotation #####################
        extend_curr_rot = quat_mul(_status._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)

        _status._rigid_body_rot = torch.cat([_status._rigid_body_rot, extend_curr_rot], dim=1)
        ################### EXTEND Rigid Body Angular Velocity #####################
        _status._rigid_body_ang_vel = torch.cat([_status._rigid_body_ang_vel, _status._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)

        ################### EXTEND Rigid Body Linear Velocity #####################
        _rigid_body_ang_vel_global = self.task.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
        angular_velocity_contribution = torch.cross(_rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        _extend_curr_vel = _status._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        _status._rigid_body_vel = torch.cat([_status._rigid_body_vel, _extend_curr_vel], dim=1)


    @property
    def num_bodies(self):
        if "extend_config" not in self.config.robot.motion:
            return 0

        return self.num_extend_bodies
