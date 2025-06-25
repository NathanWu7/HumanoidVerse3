from humanoidverse.utils.torch_utils import to_torch, get_axis_params, quat_rotate_inverse
from tools.isaac_utils.rotations import get_euler_xyz_in_tensor
from humanoidverse.envs.base_task.term import base
import torch
from loguru import logger

class TerrainStatus(base.BaseManager):

    def __init__(self, _task):
        super(TerrainStatus, self).__init__(_task)

    # stage 1
    def init(self):
        cfg = self.config.terrain
        if not cfg.measure_heights:
            return

        y = torch.tensor(cfg.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        _num_height_points = grid_x.numel()
        points = torch.zeros(self.config.num_envs, _num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        self.measure_height_points = points

    # stage 3
    def pre_compute(self):
        cfg = self.config.terrain
        if not cfg.measure_heights:
            return

        num_height_points = len(cfg.measured_points_x) * len(cfg.measured_points_y)

        if cfg.mesh_type == 'plane':
            self.env_terrain_heights = torch.zeros(self.config.num_envs, num_height_points,\
                                                   device=self.device, requires_grad=False)
        elif cfg.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        else:
            if not hasattr(self, "heightsamples"):
                terrain = self.task.simulator.terrain
                self.heightsamples = torch.tensor(terrain.heightsamples).flatten().to(self.device)
                self.tot_cols = terrain.tot_cols
                self.tot_rows = terrain.tot_rows

            robotstatus_manager = self.task.robotstatus_manager

            from humanoidverse.utils import math
            _quat = robotstatus_manager.base_quat.clone().repeat(1, num_height_points)
            _results = math.quat_apply_yaw(_quat, self.measure_height_points)
            _postions = self.task.simulator.robot_root_states[:, :3].clone().unsqueeze(1)
            measure_points = _results + _postions

            measure_points += cfg.border_size
            measure_points = (measure_points / cfg.horizontal_scale).int()

            _px = measure_points[:, :, 0].flatten()
            _py = measure_points[:, :, 1].flatten()

            _px = torch.clip(_px, 0, self.tot_cols - 2)
            _py = torch.clip(_py, 0, self.tot_rows - 2)

            _p1 = self.tot_cols * _py + _px
            _p2 = self.tot_cols * _py + _px + 1
            _p3 = self.tot_cols * (_py + 1) + _px

            heights1 = self.heightsamples[_p1]
            heights2 = self.heightsamples[_p2]
            heights3 = self.heightsamples[_p3]

            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            robotstatus_manager.env_terrain_heights = heights.view(self.config.num_envs, -1) * cfg.vertical_scale

