import numpy as np
import torch
from humanoidverse.envs.base_task.term import base

class BaseTerrainManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseTerrainManager, self).__init__(_task)
        self.hasinit_reset = False

    def pre_init(self):
        """
        Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.

        simulator had instanced
        """
        if self.config.terrain.mesh_type in ["heightfield", "trimesh"]:
            # import ipdb; ipdb.set_trace()
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.config.terrain.max_init_terrain_level
            max_init_level = self.config.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.config.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.config.terrain.num_rows
            if isinstance(self.task.simulator.terrain.env_origins, np.ndarray):
                self.terrain_origins = torch.from_numpy(self.task.simulator.terrain.env_origins).to(self.device).to(torch.float)
            else:
                self.terrain_origins = self.task.simulator.terrain.env_origins.to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            # import ipdb; ipdb.set_trace()
            # print(self.terrain_origins.shape)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.config.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    ## called by robotdata.py  RandResetDataManager
    def compute_reset_origins(self, env_ids):
        if 0 == len(env_ids) or \
           not self.custom_origins or \
           not self.config.terrain.curriculum:
            return

        if not self.hasinit_reset:
            self.hasinit_reset = True
            return

        commands = self.task.command_manager.commands
        # compute the distance the robot walked
        distance = torch.norm(self.task.simulator.robot_root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terrains
        move_up = distance > self.config.terrain.terrain_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = distance < torch.norm(commands[env_ids, :2], dim=1) * self.config.max_episode_length_s * 0.5
        move_down *= ~move_up
        # update terrain levels

        # update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        # update the env origins
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def post_compute(self):
        if not self.custom_origins or \
           not self.config.terrain.curriculum:
            return

        assert hasattr(self.task, "extras_manager")
        extras_manager = self.task.extras_manager
        extras_manager.log_dict["terrain_levels"] = self.terrain_levels.float()
