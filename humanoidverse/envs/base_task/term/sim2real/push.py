from humanoidverse.envs.base_task.term import base
from humanoidverse.utils.torch_utils import torch_rand_float
import torch

# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

class PushManager(base.BaseManager):

    def __init__(self, _task):
        super(PushManager, self).__init__(_task)
        if not self.config.domain_rand.push_robots:
            return

        self.push_interval_s = torch.randint(self.config.domain_rand.push_interval_s[0],
                                             self.config.domain_rand.push_interval_s[1], (self.num_envs,), device=self.device)

    # stage 1
    def init(self):
        if not self.config.domain_rand.push_robots:
            return
        self.push_robot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.push_robot_plot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)


    #def init_domain_rand(self):
    def post_init(self):
        if not self.config.domain_rand.push_robots:
            return
        self.push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.record_push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

    # stage 3
    #def pre_compute_observations(self):
    def pre_compute(self):
        if not self.config.domain_rand.push_robots:
            return
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1

        push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.task.dt).int()).nonzero(as_tuple=False).flatten()
        self.push_robot_counter[push_robot_env_ids] = 0
        self.push_robot_plot_counter[push_robot_env_ids] = 0
        self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], \
                                                                 self.config.domain_rand.push_interval_s[1], \
                                                                 (len(push_robot_env_ids),), device=self.device, requires_grad=False)
        self._push_robots(push_robot_env_ids)

    def _push_robots(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if len(env_ids) == 0:
            return
        self.task.need_to_refresh_envs[env_ids] = True

        max_vel = self.config.domain_rand.max_push_vel_xy
        self.push_robot_vel_buf[env_ids] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=str(self.device))  # lin vel x/y
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.task.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.task.simulator.all_root_states))

    # stage 4
    def draw_debug_vis(self):
        if not self.config.domain_rand.push_robots:
            return
        # draw push robot
        draw_env_ids = (self.push_robot_plot_counter < 10).nonzero(as_tuple=False).flatten()
        not_draw_env_ids = (self.push_robot_plot_counter >= 10).nonzero(as_tuple=False).flatten()
        self.record_push_robot_vel_buf[not_draw_env_ids] *=0
        self.push_robot_plot_counter[not_draw_env_ids] = 0

        for env_id in draw_env_ids:
            push_vel = self.record_push_robot_vel_buf[env_id]
            push_vel = torch.cat([push_vel, torch.zeros(1, device=self.device)])
            push_pos = self.task.simulator.robot_root_states[env_id, :3]
            push_vel_list = [push_vel]
            push_pos_list = [push_pos]
            push_mag_list = [1]
            push_color_schems = [(0.851, 0.144, 0.07)]
            push_line_widths = [0.03]
            for push_vel, push_pos, push_mag, push_color, push_line_width in zip(push_vel_list, push_pos_list, push_mag_list, push_color_schems, push_line_widths):
                for _ in range(200):
                    self.task.simulator.draw_line(Point(push_pos +torch.rand(3, device=self.device) * push_line_width),
                                        Point(push_pos + push_vel * push_mag),
                                        Point(push_color),
                                        env_id)