import torch
from humanoidverse.envs.base_task.term import base

class BaseActionsManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseActionsManager, self).__init__(_task)
        self.dim_actions = self.task.config.robot.actions_dim

    # stage 1
    def init(self):
        self.obs_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.compute_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)

    # stage 2
    def pre_physics_step(self, actions):
        self.obs_actions = actions
        clip_action_limit = self.config.robot.control.action_clip_value
        self.compute_actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

    def post_physics_step(self):
        assert hasattr(self.task, "extras_manager")
        _log_dict = self.extras_manager.log_dict
        clip_action_limit = self.config.robot.control.action_clip_value

        _log_dict["action_clip_frac"] = (
                self.compute_actions.abs() == clip_action_limit
            ).sum() / self.compute_actions.numel()

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.obs_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.

    def post_step(self):
        self.last_actions[:] = self.obs_actions[:]

    @property
    def actual_actions(self):
        return self.compute_actions

    ######################### Observations #########################
    def _get_obs_actions(self,):
        return self.obs_actions


    def zeros(self):
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        return actor_state

    def check(self, num_dof):
        assert num_dof == self.dim_actions, "Number of DOFs must be equal to number of actions"


from humanoidverse.envs.base_task.actions import randomize_ctrl_delay
class ActionsManager(BaseActionsManager):
    def __init__(self, _task):
        super(ActionsManager, self).__init__(_task)
        self.ctrl_delay = randomize_ctrl_delay.CtrlDelayManager(_task)

    # stage 1
    def init(self):
        super(ActionsManager, self).init()
        self.ctrl_delay.init()

    # stage 2
    def pre_physics_step(self, actions):
        super(ActionsManager, self).pre_physics_step(actions)
        ## delay
        self.compute_actions = self.ctrl_delay.pre_physics_step(self.compute_actions)

    # stage 3
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        super(ActionsManager, self).reset(env_ids)
        ## delay
        self.ctrl_delay.reset(env_ids)
