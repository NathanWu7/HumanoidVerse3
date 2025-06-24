from humanoidverse.envs.base_task.term import base
from humanoidverse.utils.torch_utils import quat_apply, to_torch, torch_rand_float
from isaac_utils.rotations import wrap_to_pi
import torch

class VelocityCommand(base.BaseManager):
    def __init__(self, _task):
        super(VelocityCommand, self).__init__(_task)

    # stage 1
    def init(self):
        super(VelocityCommand, self).init()
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.commands = torch.zeros((self.num_envs, 4),
                                     dtype=torch.float32,
                                     device=self.device)

        self.command_ranges = self.config.locomotion_command_ranges

    # stage 3
    def pre_compute(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        # commands
        self.command_counter[:] += 1
        episode_manager = self.task.episode_manager
        robotdata_manager = self.task.robotdata_manager
        robotstatus_manager = self.task.robotstatus_manager

        if True: #not self.task.is_evaluating:
            env_ids = (episode_manager.episode_length_buf % \
                       int(self.config.locomotion_command_resampling_time / self.task.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample(env_ids)

        forward = quat_apply(robotstatus_manager.base_quat, robotdata_manager.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading),
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1]
        )

    def reset(self, env_ids):
        if 0 == len(env_ids):
            return
        self._resample(env_ids)

    def post_compute(self):
        if self.task.headless:
            return
        # for debug show
        self.task.simulator.commands = self.commands

    def _resample(self, env_ids):
        if 0 == len(env_ids):
            return

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # help function
    def update(self, command):
        command = torch.tensor(command, dtype=torch.float32, device = self.device)
        _shape = command.shape[-1]
        self.commands[:, :_shape] =  command # only set the first 3 commands

    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]

    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]
