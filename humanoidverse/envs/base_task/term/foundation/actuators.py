import torch
from humanoidverse.utils.torch_utils import torch_rand_float
from humanoidverse.envs.base_task.term import base
from loguru import logger

class ActuatorsManager(base.BaseManager):
    def __init__(self, _task):
        super(ActuatorsManager, self).__init__(_task)
        self.dim_actions = self.task.config.robot.actions_dim

    # stage 1
    def init(self):
        self.torques = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)

    #def init_domain_rand(self):
    def post_init(self):
        ### only for robotdata_manager init
        robotdata_manager = self.task.robotdata_manager

        for i in range(robotdata_manager.num_dof):
            name = robotdata_manager.dof_names[i]

            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.p_gains[i]} and {self.d_gains[i]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.config.robot.control.control_type in ["P", "V"]:
                    logger.warning(f"PD gain of joint {name} were not defined, setting them to zero")
                    raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        ###
        self._kp_scale = torch.ones(self.num_envs, robotdata_manager.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, robotdata_manager.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, robotdata_manager.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    # stage 2
    def pre_physics_step(self, actions):
        ## _compute_torques
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # actions *= 0.
        # print("self.task.simulator.dof_vel", self.task.simulator.dof_vel)
        # print("actions", actions)
        robotdata_manager = self.task.robotdata_manager
        task = self.task

        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains * (actions_scaled + robotdata_manager.default_dof_pos - task.simulator.dof_pos) - \
                      self._kd_scale * self.d_gains * task.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains * (actions_scaled - task.simulator.dof_vel) - \
                      self._kd_scale * self.d_gains * (task.simulator.dof_vel - self.last_dof_vel) / self.sim_dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques) * 2. - 1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * robotdata_manager.torque_limits

        if self.config.robot.control.clip_torques:
            _torques = torch.clip(torques, - robotdata_manager.torque_limits, robotdata_manager.torque_limits)
        else:
            _torques = torques

        self.torques = _torques.view(self.torques.shape)

    # stage 3
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        robotdata_manager = self.task.robotdata_manager
        if self.config.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.config.domain_rand.kp_range[0],
                                                       self.config.domain_rand.kp_range[1],
                                                       (len(env_ids), robotdata_manager.num_dof), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.config.domain_rand.kd_range[0],
                                                       self.config.domain_rand.kd_range[1],
                                                       (len(env_ids), robotdata_manager.num_dof), device=self.device)

        if self.config.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.config.domain_rand.rfi_lim_range[0],
                                                            self.config.domain_rand.rfi_lim_range[1],
                                                            (len(env_ids), robotdata_manager.num_dof), device=self.device)
