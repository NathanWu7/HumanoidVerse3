import torch
from humanoidverse.utils.torch_utils import to_torch
from humanoidverse.envs.base_task.term import base
from loguru import logger
from humanoidverse.utils.torch_utils import torch_rand_float
from easydict import EasyDict

class BaseRobotDataManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseRobotDataManager, self).__init__(_task)

    def preinstance_task_data(self):
        self.task.task_data = EasyDict()
        self.task_data = self.task.task_data

    def pre_init(self):
        '''
        simulator had instanced
        '''
        logger.info("TODO  Note don\'t know crash")
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        logger.info("TODO  Note don\'t know crash")

        self.num_dof = self.task.simulator.num_dof
        self.num_bodies = self.task.simulator.num_bodies
        self.dof_names = self.task.simulator.dof_names
        self.body_names = self.task.simulator.body_names
        # check dimensions
        if hasattr(self.task, "actions_manager"):
            self.task.actions_manager.check(self.num_dof)

        # other properties
        self.num_bodies = len(self.body_names)
        #self.num_dofs = len(self.dof_names)
        assert len(self.dof_names) == self.num_dof
        base_init_state_list = self.config.robot.init_state.pos + \
                               self.config.robot.init_state.rot + \
                               self.config.robot.init_state.lin_vel + \
                               self.config.robot.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def init(self):
        '''
        envs had created
        '''
        ## limits
        self.dof_pos_limits, \
        self.dof_vel_limits, \
        self.torque_limits = self.task.simulator.get_dof_limits_properties()

        ## pos
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self._setup_robot_body_indices()

    def _setup_robot_body_indices(self):
        feet_names = [s for s in self.body_names if self.config.robot.foot_name in s]
        knee_names = [s for s in self.body_names if self.config.robot.knee_name in s]
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.task.simulator.find_rigid_body_indice(feet_names[i])

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.task.simulator.find_rigid_body_indice(knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.task.simulator.find_rigid_body_indice(penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.task.simulator.find_rigid_body_indice(termination_contact_names[i])

        if self.config.robot.has_upper_body_dof:
            # maintain upper/lower dof idxs
            self.upper_dof_names = self.config.robot.upper_dof_names
            self.lower_dof_names = self.config.robot.lower_dof_names
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]

        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.task.simulator.find_rigid_body_indice(self.torso_name)

        ### TODO
        if self.config.robot.get("upper_left_arm_dof_names", None):
            self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
            self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
            self.upper_left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
            self.upper_right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]

        if self.config.robot.motion.get("hips_link", None):
            self.hips_dof_id = [self.task.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
            logger.info("Yuanhang: -1 for the base link (pelvis)")


class RandResetDataManager(BaseRobotDataManager):
    def __init__(self, _task):
        super(RandResetDataManager, self).__init__(_task)

    # stage 3
    def reset(self, env_ids):
        super(RandResetDataManager, self).reset(env_ids)
        if len(env_ids) == 0:
            return

        self.task.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
        self.task.simulator.dof_vel[env_ids] = 0.

        # base position
        terrain_manager = self.task.terrain_manager
        if terrain_manager.custom_origins:
            terrain_manager.compute_reset_origins(env_ids)
            self.task.simulator.robot_root_states[env_ids] = self.base_init_state
            self.task.simulator.robot_root_states[env_ids, :3] += terrain_manager.env_origins[env_ids]
            self.task.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
        else:
            self.task.simulator.robot_root_states[env_ids] = self.base_init_state
            self.task.simulator.robot_root_states[env_ids, :3] += terrain_manager.env_origins[env_ids]

        # base velocities
        self.task.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel

