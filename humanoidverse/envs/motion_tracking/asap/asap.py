from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.envs.motion_tracking.asap import register
from humanoidverse.envs.motion_tracking.asap.extends import asap_motion_save
from hydra.utils import instantiate

class ASAPMotionTrainer(BaseTask):
    def __init__(self, config, device):
        super(ASAPMotionTrainer, self).__init__(config, device)

    def _pre_init(self):
        super()._pre_init()
        self.motion_lib = instantiate(self.config.robot.motion, device=self.device)

    @property
    def manager_map(self):
        return register.asap_trainer_registry

    @property
    def rewards_map(self):
        return register.asap_rewards_registry


class ASAPMotionEvaluater(BaseTask):
    def __init__(self, config, device):
        super(ASAPMotionEvaluater, self).__init__(config, device)
        self.debug_viz = True

    def next_task(self):
        self.robotdata_manager.next_task()

    @property
    def manager_map(self):
        return register.asap_evaluater_registry

    @property
    def rewards_map(self):
        return {}

class ASAPMotionPlayer(ASAPMotionEvaluater):
    def __init__(self, config, device):
        super(ASAPMotionPlayer, self).__init__(config, device)

    def _reset(self, _env_ids = None):
        # update every envs
        import torch
        _env_ids = torch.arange(self.num_envs, device=self.device)
        super(ASAPMotionPlayer, self)._reset(_env_ids)

    @property
    def manager_map(self):
        return register.asap_evaluater_registry
