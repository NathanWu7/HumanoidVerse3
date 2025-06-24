from agents.base_algo import base
from agents.base_algo import base_algo
import torch

class Rollout(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Rollout, self).__init__(_algo)


    # stage 3
    #def pre_step(self, _flags: int = -1):
    #    pass

    def step(self):
        # Compute the actions and values
        # actions = self.actor.act(obs_dict["actor_obs"]).detach()
        _inputs = {'context': base.Context.ROLLOUT.value}

        # stage 1
        # 1 feature obs_dict
        self.algo.envwarp_component.pre_step(_inputs)

        # stage 2
        # 2 feature policy_state_dict
        self.algo.modules_component.step(_inputs)
        # 3 append states to storage
        self.algo.storage_component.step(_inputs)
        # 4 env update
        self.algo.envwarp_component.step(_inputs)

        # stage 3
        # 5 append rewards, dones to storage
        self.algo.storage_component.post_step(_inputs)
        # 6 # Book keeping
        self.algo.statistics_component.post_step(_inputs)
        # reset dones for relu or rnn
        self.algo.modules_component.post_step(_inputs)

    def pre_epoch(self):
        _inputs = {'context': base.Context.ROLLOUT.value}
        # clear storage
        self.algo.storage_component.pre_epoch(_inputs)

    def post_epoch(self):
        # prepare data for training
        _inputs = {'context': base.Context.ROLLOUT.value}
        # stage 1
        # 1 feature obs_dict
        self.algo.envwarp_component.pre_step(_inputs)
        # 2. Compute the returns and advantages
        self.algo.storage_component.post_epoch(_inputs)

