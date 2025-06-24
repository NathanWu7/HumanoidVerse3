from humanoidverse.envs.base_task.term import base
import torch

class CtrlDelayManager(base.BaseManager):
    def __init__(self, _task):
        super(CtrlDelayManager, self).__init__(_task)
        self.dim_actions = self.task.config.robot.actions_dim

    # stage 1
    def init(self):
        if not self.config.domain_rand.randomize_ctrl_delay:
            return

        self.action_queue = torch.zeros(self.num_envs,
                                        self.config.domain_rand.ctrl_delay_step_range[1]+1,
                                        self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.action_delay_idx = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0],
                                            self.config.domain_rand.ctrl_delay_step_range[1]+1,
                                            (self.num_envs,), device=self.device, requires_grad=False)

    def pre_physics_step(self, actions):
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions
            _actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            _actions_after_delay = actions.clone()

        return _actions_after_delay

    def reset(self, env_ids):
        if not self.config.domain_rand.randomize_ctrl_delay:
            return
        if len(env_ids) == 0:
            return

        # self.action_queue[env_ids] = 0.delay:
        self.action_queue[env_ids] *= 0.
        # self.action_queue[env_ids] = 0.
        self.action_delay_idx[env_ids] = torch.randint(
                self.config.domain_rand.ctrl_delay_step_range[0],
                self.config.domain_rand.ctrl_delay_step_range[1]+1,
                    (len(env_ids),), device=self.device, requires_grad=False)
