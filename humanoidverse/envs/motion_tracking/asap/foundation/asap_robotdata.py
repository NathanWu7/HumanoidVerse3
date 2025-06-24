import torch
import numpy as np
from humanoidverse.envs.base_task.term.foundation import robotdata
from humanoidverse.envs.motion_tracking.asap.motions import aosp_motion
from easydict import EasyDict
import hydra

def _small_random_quaternions(obj, n, max_angle):
    axis = torch.randn((n, 3), device=obj.device)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
    angles = max_angle * torch.rand((n, 1), device=obj.device)

    # Convert angle-axis to quaternion
    sin_half_angle = torch.sin(angles / 2)
    cos_half_angle = torch.cos(angles / 2)

    q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)
    return q

class AsapRobotData(robotdata.BaseRobotDataManager):
    def __init__(self, _task):
        super(AsapRobotData, self).__init__(_task)
        self.preinstance_motion()

    def preinstance_motion(self):
        self.motion_warp = aosp_motion.AsapMotion(self.task)

    def pre_init(self):
        super(AsapRobotData, self).pre_init()
        self.motion_warp.pre_init()

    def init(self):
        super().init()
        self.is_initialized = True
        if "extend_config" in self.config.robot.motion:
            for extend_config in self.config.robot.motion.extend_config:
                self.task.simulator._body_list.append(extend_config["joint_name"])

        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.task.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.task.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.task.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]

    def pre_compute(self):
        super(AsapRobotData, self).pre_compute()
        self.motion_warp.pre_compute()

    def reset(self, env_ids):
        super(AsapRobotData, self).reset(env_ids)
        self.motion_warp.reset(env_ids)
        if len(env_ids) != 0:
            # only for reset motion res
            _motion_ref = self.motion_warp.motion_ref

            ## stage 2
            self._reset_dofs(env_ids, _motion_ref)
            self._reset_root_states(env_ids, _motion_ref)
        ##
        self.next_motion_ref = self.motion_warp.next_motion_ref

    def post_compute(self):
        super(AsapRobotData, self).post_compute()
        self.current_motion_ref = self.next_motion_ref

    def _reset_dofs(self, env_ids, _motion_ref):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        motion_ref = _motion_ref

        dof_pos = motion_ref['dof_pos'][env_ids]
        dof_vel = motion_ref['dof_vel'][env_ids]
        if self.task.is_evaluating:
            self.task.simulator.dof_pos[env_ids] = dof_pos
            self.task.simulator.dof_vel[env_ids] = dof_vel
            return

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        self.task.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.task.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise

    def _reset_root_states(self, env_ids, _motion_ref):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        assert hasattr(self.task, "terrain_manager")
        terrain_manager = self.task.terrain_manager
        motion_ref = _motion_ref

        if terrain_manager.custom_origins:

            self.task.simulator.robot_root_states[env_ids, :3] = motion_ref['root_pos'][env_ids]
            # self.robot_root_states[env_ids, 2] += 0.04 # in case under the terrain
            if self.config.simulator.config.name == 'isaacgym':
                self.task.simulator.robot_root_states[env_ids, 3:7] = motion_ref['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim' or self.config.simulator.config.name == 'isaacsim45':
                from isaac_utils.rotations import xyzw_to_wxyz
                self.task.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_ref['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.task.simulator.robot_root_states[env_ids, 3:7] = motion_ref['root_rot'][env_ids]
                raise NotImplementedError
            self.task.simulator.robot_root_states[env_ids, 7:10] = motion_ref['root_vel'][env_ids]
            self.task.simulator.robot_root_states[env_ids, 10:13] = motion_ref['root_ang_vel'][env_ids]

        else:
            from isaac_utils.rotations import quat_mul, xyzw_to_wxyz

            if self.task.is_evaluating:
                root_pos_noise = 0
                root_rot_noise = 0
                root_vel_noise = 0
                root_ang_vel_noise = 0
            else:
                root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
                root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
                root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
                root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_ref['root_pos'][env_ids]
            root_rot = motion_ref['root_rot'][env_ids]
            root_vel = motion_ref['root_vel'][env_ids]
            root_ang_vel = motion_ref['root_ang_vel'][env_ids]

            self.task.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.task.simulator.robot_root_states[env_ids, 3:7] = quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim' or self.config.simulator.config.name == 'isaacsim45':
                self.task.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.task.simulator.robot_root_states[env_ids, 3:7] = quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.task.simulator.robot_root_states[env_ids, 3:7] = quat_mul(_small_random_quaternions(self, root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            else:
                raise NotImplementedError
            self.task.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.task.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise

    @property
    def motion_start_times(self):
        return self.motion_warp.motion_start_times

    @property
    def next_motion_times(self):
        return self.motion_warp.motion_start_times

    @property
    def motion_len(self):
        return self.motion_warp.motion_len


class AsapRobotDataEvaluater(AsapRobotData):
    def __init__(self, _task):
        super(AsapRobotDataEvaluater, self).__init__(_task)

    def preinstance_motion(self):
        self.motion_warp = aosp_motion.AsapMotionEvaluater(self.task)

    def next_task(self):
        self.motion_warp.next_task()

    def draw_debug_vis(self):
        robotdata_manager = self
        if not hasattr(robotdata_manager, "current_motion_ref"):
            return

        ref_body_pos = robotdata_manager.current_motion_ref["rg_pos_t"]
        ref_body_pos = ref_body_pos.reshape(self.num_envs, -1, 3)

        for env_id in range(self.num_envs):

            # draw marker joints
            for pos_id, pos_joint in enumerate(ref_body_pos[env_id]): # idx 0 torso (duplicate with 11)
                if self.config.robot.motion.visualization.customize_color:
                    _id = pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)
                    color_inner = self.config.robot.motion.visualization.marker_joint_colors[_id]
                else:
                    color_inner = (0.3, 0.3, 0.3)
                color_inner = tuple(color_inner)

                self.task.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)


class AsapRobotDataPlayer(AsapRobotDataEvaluater):
    def __init__(self, _task):
        super(AsapRobotDataPlayer, self).__init__(_task)

    def reset(self, env_ids):
        # only for reset motion res
        if hasattr(self, 'current_motion_ref'):
            _motion_ref = self.current_motion_ref
        else:
            _motion_ref = self.motion_warp.motion_ref

        ## stage 2
        self._reset_dofs(env_ids, _motion_ref)
        self._reset_root_states(env_ids, _motion_ref)
        self.next_motion_ref = self.motion_warp.next_motion_ref

        ## for debug
        #_ref_motion_length = self.motion_warp.motion_len
        #_ref_motion_phase = self.motion_warp._next_motion_times / _ref_motion_length
        #print("motion_phase", _ref_motion_phase)
